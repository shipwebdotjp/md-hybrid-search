# md-hybrid-search 仕様

## 1. 目的

このライブラリは、Obsidian を中心とした Markdown ベースの検索ワークフローのために、以下だけを担当する。

- 指定フォルダ群を再帰的にスキャンする
- 追加・変更・削除を検出する
- Markdown をチャンク化する
- Embedding を生成する
- SQLite FTS5 による BM25 検索を維持する
- ChromaDB にベクトルを保存する
- BM25 と similarity search を統合して検索する

公開 API は最小限に絞り、外部アプリからは主に `sync()` と `search()` を使う。

### 非対象

- Web クロール
- AI による要約や回答生成
- DeepResearch
- ナレッジグラフ構築
- Web UI
- コレクション横断検索の上位 API

### v1 の方針

- embedding provider は内包しない
- tokenizer は内部実装とし、public API では原則公開しない
- chunking は固定長分割を基本とする
- `sync()` 失敗時は例外を握りつぶさない
- embedding cache は提供しない
- 自動 migration は提供しない

---

## 2. 主要概念

### 2.1 collection

`collection_name` は 1 つの論理インデックスを表す。

- 1 collection = 1 directory ではない
- 1 collection に複数の `DirectorySource` を含められる
- SQLite と Chroma はアプリ全体で共有し、`collection_name` で分割する
- v1 では `collection_name` を ChromaDB の physical collection name としても使う

collection_name は次を満たす必要がある。

- 先頭と末尾は英数字
- 使用可能文字は英数字、underscore、hyphen
- 長さは 3 文字以上 63 文字以下を推奨
- 空文字は不可

不正な `collection_name` が指定された場合、`ValueError` を送出する。

### 2.2 source

source は collection に属する入力元ディレクトリを表す。
v1 では `DirectorySource` のみを公開し、Markdown を再帰スキャンする。

source はフォルダ単位の設定であり、collection の分割単位ではない。

### 2.3 chunk

chunk は検索と embedding の最小単位である。

- 1 file は 1 つ以上の chunk に分割される
- chunk 単位で SQLite と Chroma に保存する
- chunk_id は collection 内で一意でなければならない

chunk_id は次の情報から決定的に生成する。

- `collection_name`
- `file_path`
- `chunk_index`
- `content_hash`

---

## 3. 公開 API

### 3.1 `DirectorySource`

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class DirectorySource:
    path: str
```

#### 仕様

- `path` は必須
- `path` はライブラリ側で正規化する
- v1 は Markdown 専用とする
- v1 は再帰スキャン固定とする
- include / exclude pattern は公開しない

#### 意図

`DirectorySource` は単純な値オブジェクトにする。
外部アプリは「どのディレクトリを索引対象にするか」だけを渡せばよい。

### 3.2 `SearchIndex`

```python
index = SearchIndex(
    collection_name="main",
    sources=[
        DirectorySource("/path/to/vault"),
        DirectorySource("/path/to/another-dir"),
    ],
    sqlite_path="/Users/me/AppData/obsidian-ai-hub/search.sqlite",
    chroma_path="/Users/me/AppData/obsidian-ai-hub/chroma",
    embedder=embedder,
    chunk_size=1000,
    chunk_overlap=100,
)
```

#### コンストラクタの責務

- collection 名と source 群を受け取る
- SQLite と Chroma の永続化先を受け取る
- embedder を受け取る
- reranker を受け取る
- chunking 設定を受け取る
- 内部で SQLite スキーマと Chroma collection を準備する

#### コンストラクタに含めないもの

- `vector_store` オブジェクト
- crawler 設定
- AI 回答設定
- DeepResearch 設定

v1 では、ライブラリが `chroma_path` から `PersistentClient` を生成し、collection を自前で管理する。
外部アプリは vector store 実装を直接渡さない。

### 3.3 `sync()`

```python
report = index.sync()
```

#### 仕様

`sync()` は collection に属する全 source を走査し、差分を反映する。

#### 処理順

1. source 一覧を正規化する
2. 各 source が存在することを確認する
3. 各 source 配下の Markdown ファイルを再帰スキャンする
4. 現在のファイル一覧と SQLite 上の manifest を比較する
5. 新規・変更ファイルは既存 chunk を削除して再作成する
6. 削除ファイル、または collection から外れた source のファイルは削除する
7. chunk を SQLite FTS5 と ChromaDB の両方へ保存する
8. collection metadata を更新する

#### `sync()` の戻り値

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class SyncReport:
    collection_name: str
    scanned_files: int
    new_files: int
    updated_files: int
    unchanged_files: int
    deleted_files: int
    inserted_chunks: int
    deleted_chunks: int
```

#### 仕様

- `sources` が空の場合、`sync()` は `ValueError` を送出する
- source path が存在しない場合、`sync()` は `FileNotFoundError` を送出する
- source path が存在しない場合、その source 配下の既存 index は削除しない
- Markdown 以外は対象外
- ファイルの変更検知は `mtime` と `size` と `content_hash` を使う
- `mtime` / `size` が一致し、`content_hash` も一致する場合は unchanged とする
- `mtime` / `size` が変わった場合は `content_hash` を計算して変更判定する
- 変更ファイルはファイル単位で chunk を削除して再投入する
- sync 中のエラーは握りつぶさず、呼び出し元へ例外として伝播する
- SQLite 更新は transaction 内で行う
- SQLite と ChromaDB は完全な atomic transaction を共有できないため、不整合が疑われる場合は `rebuild()` を使う

### 3.4 `search()`

```python
results = index.search(
    query="検索クエリ",
    limit=10,
    mode="hybrid",
)
```

#### 仕様

- `query` が空文字または空白のみの場合は `ValueError`
- `limit` は 1 以上
- `mode` が不正な場合は `ValueError`
- collection 内のみを検索対象にする
- 返却件数は最大 `limit`
- `score` は大きいほど上位を表す

#### 戻り値

```python
from dataclasses import dataclass
from typing import Any, Literal

@dataclass(frozen=True)
class SearchHit:
    chunk_id: str
    score: float
    mode: Literal["keyword", "similarity", "hybrid"]
    content: str
    metadata: dict[str, Any]
```

#### 結果の性質

- `content` は検索対象 chunk の本文
- `metadata` には少なくとも以下を含める
  - `collection_name`
  - `source_path`
  - `file_path`
  - `relative_path`
  - `chunk_index`
  - `mtime`
  - `content_hash`
- `mode` は `keyword` / `similarity` / `hybrid` のいずれか

---

## 4. 保存先と collection の扱い

### 4.1 保存先

外部アプリが保存場所を決める。

```text
app-data/
  search.sqlite
  chroma/
```

### 4.2 役割分担

#### 呼び出し元アプリ

- `app_data_dir` を決める
- `sqlite_path` を決める
- `chroma_path` を決める
- `collection_name` を決める
- `sources` を保存する
- API key を保存する
- embedder を生成する

#### md-hybrid-search

- SQLite に接続する
- schema を作る
- collection metadata を保存する
- file / chunk / FTS を更新する
- Chroma PersistentClient を作る
- Chroma collection を作る
- chunk_id 単位で upsert / delete する
- hybrid search を実行する

### 4.3 Chroma collection

- v1 では `collection_name` ごとに ChromaDB の collection を作成する
- `collection_name` は logical name であり、physical name と一致させる
- 不正な `collection_name` は `ValueError` とする
- Chroma の ID は `chunk_id` を使う
- SQLite 側も Chroma 側も `chunk_id` を正とする

Chroma に保存する metadata は SQLite の chunks と揃える。

```python
metadata = {
    "collection_name": collection_name,
    "source_path": source_path,
    "file_path": file_path,
    "relative_path": relative_path,
    "chunk_index": chunk_index,
    "mtime": mtime,
    "content_hash": content_hash,
}
```

```python
collection.upsert(
    ids=[chunk_id],
    documents=[content],
    embeddings=[embedding],
    metadatas=[metadata],
)
```

---

## 5. 内部データモデル

公開 API では隠すが、実装上は次の概念を持つ。

### 5.1 collections

- `collection_name`
- `created_at`
- `updated_at`
- `metadata_json`

`metadata_json` には少なくとも以下を含める。

- `chunk_size`
- `chunk_overlap`
- `embedder_fingerprint`
- `tokenizer_fingerprint`
- `schema_version`

### 5.2 sources

- `collection_name`
- `source_path`
- `created_at`
- `updated_at`

### 5.3 files

- `collection_name`
- `file_path`
- `source_path`
- `relative_path`
- `mtime`
- `size`
- `content_hash`
- `last_indexed_at`

### 5.4 chunks

- `chunk_id`
- `collection_name`
- `file_path`
- `source_path`
- `relative_path`
- `chunk_index`
- `content`
- `content_hash`
- `token_count`
- `mtime`
- `created_at`

### 5.5 schema_meta

- `key`
- `value`

schema version などの管理に使う。

### 5.6 chunks_fts

v1 では FTS5 は duplicate storage としてよい。

- `chunk_id`
- `collection_name`
- `content`

`chunks.content` には元の chunk 本文を保存し、`chunks_fts.content` には検索用に正規化した text を保存する。

### 5.7 chunk_id の考え方

chunk_id は collection 内で一意である必要がある。
v1 では次の情報を材料にした決定的 ID を使う。

- `collection_name`
- `file_path`
- `chunk_index`
- `content_hash`

これにより、ファイル単位の削除と再投入が簡単になる。

---

## 6. Embedder インターフェース

`md-hybrid-search` は embedding provider を内包しない。
外部アプリは以下の protocol を満たす embedder を渡す。

```python
from typing import Protocol

class Embedder(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        ...

    def embed_query(self, text: str) -> list[float]:
        ...
```

### 仕様

- `embed_documents()` は sync 時に chunk 本文の embedding を生成するために使う
- `embed_query()` は similarity search 時に query embedding を生成するために使う
- collection 内の embedding dimension は一定でなければならない
- embedder を変更した場合、既存 index の再構築が必要
- embedding 生成に失敗した場合、`sync()` は例外を送出する
- v1 では embedding cache は提供しない

---

## 7. Tokenizer

SQLite FTS5 に保存する text は、内部 tokenizer により検索用に正規化される。

v1 では tokenizer を内部実装とし、public API では原則公開しない。
ただし、tokenizer 変更時は index rebuild が必要であることを仕様として定める。

### v1 の方針

- 日本語向け tokenizer が利用可能な場合はそれを使う
- 利用できない場合は simple regex tokenizer に fallback する
- tokenizer の違いにより keyword 検索結果が変わることは許容する
- tokenizer の変更後は `rebuild()` が必要

### FTS に保存する値

- `chunks.content` には元の chunk 本文を保存する
- FTS5 table には tokenizer 適用後の text を保存する
- 検索 query にも同じ tokenizer を適用する

---

## 8. Path normalization

`DirectorySource.path` は次の手順で正規化する。

1. `~` を展開する
2. 相対パスを絶対パスに変換する
3. filesystem 上の実体パスへ resolve する

### source の重複

同一 path の source が複数指定された場合、重複を除去する。

### 親子 source

ある source が別の source の子孫である場合、v1 では `ValueError` を送出する。

例:

```python
sources=[
    DirectorySource("/vault"),
    DirectorySource("/vault/projects"),
]
```

この場合、`/vault/projects` は `/vault` に含まれるため不正とする。

### symlink

- v1 では symlink directory は辿らない
- symlink file は実体 path に resolve して扱う

---

## 9. Chunking

v1 では Markdown をプレーンテキストとして扱い、固定長 chunk に分割する。

### デフォルト値

- `chunk_size`: 1000 characters
- `chunk_overlap`: 100 characters

### 注意

- v1 では Markdown heading に基づく semantic chunking は行わない
- v1 では YAML frontmatter を本文に含める
- v1 では code block を特別扱いしない
- chunking 設定を変更した場合は `rebuild()` が必要

### 将来検討

- heading-aware chunking
- frontmatter 除外
- line number metadata
- section title metadata

---

## 10. sync の詳細仕様

### 10.1 スキャン

- source 配下を再帰的に走査する
- 対象は `.md` のみ
- 収集するのは絶対パス

### 10.2 新規ファイル

- chunk 化する
- embedding を作る
- SQLite FTS5 に保存する
- ChromaDB に保存する
- manifest を更新する

### 10.3 変更ファイル

- 既存 chunk を削除する
- 再 chunk 化する
- 再 embedding する
- 再保存する

### 10.4 削除ファイル

- SQLite から削除する
- ChromaDB から削除する

### 10.5 source 変更

collection に登録されていた source が `sources` から外れた場合、その source に属していた file / chunk は削除対象とする。

これは collection が「現在の source 群の集合」を表すためである。

### 10.6 安全性

- `sources` が空の場合、`sync()` は `ValueError` を送出する
- source path が存在しない場合、`sync()` は `FileNotFoundError` を送出する
- source path が存在しない場合、その source 配下の既存 index は削除しない
- dry-run は提供しない

---

## 11. search の詳細仕様

### 11.1 keyword

- SQLite FTS5 で検索する
- BM25 の順位を使う
- 対象 collection のみを見る

### 11.2 similarity

- ChromaDB でベクトル検索する
- 対象 collection のみを見る

### 11.3 hybrid

- keyword と similarity の両方を実行する
- keyword candidate count = `max(limit * 5, 50)`
- similarity candidate count = `max(limit * 5, 50)`
- RRF で統合する
- RRF k は 60 とする
- 同一 `chunk_id` の結果は 1 件にまとめる
- 上位 `limit` 件を返す

### 11.4 返却

- 返却順は `score` の降順
- `query` が空文字または空白のみの場合は `ValueError`
- `limit` は 1 以上
- `mode` が不正な場合は `ValueError`

---

## 12. Search score

`SearchHit.score` は、すべての mode において「大きいほど上位」を表す。

### keyword mode

SQLite FTS5 の `bm25()` は値が小さいほど関連度が高い。
そのため、`SearchHit.score` に raw bm25 value をそのまま返さない。

v1 では keyword mode の score は順位ベースで計算する。

```python
score = 1.0 / rank
```

ここで rank は 1 始まりとする。

### similarity mode

ChromaDB の distance は metric により意味が異なるため、raw distance をそのまま返さない。
v1 では similarity mode の score も順位ベースで計算する。

```python
score = 1.0 / rank
```

### hybrid mode

hybrid mode の score は RRF score とする。

```python
score = sum(1 / (rrf_k + rank_i))
```

v1 の `rrf_k` は 60 とする。

### 注意

score は同一検索 mode 内での相対比較用であり、異なる mode 間で比較可能であることは保証しない。

---

## 13. Rebuild and Clear

### rebuild()

```python
report = index.rebuild()
```

`rebuild()` は collection の既存 index をすべて削除し、現在の sources から再構築する。

以下の場合に使用する。

- embedder を変更した
- embedding dimension が変わった
- tokenizer を変更した
- chunking 設定を変更した
- SQLite と ChromaDB の不整合が疑われる
- index を完全に作り直したい

### clear()

```python
index.clear()
```

`clear()` は collection の SQLite / ChromaDB 上の index を削除する。

`clear()` は filesystem 上の Markdown ファイルを削除しない。

### Config mismatch

collection に保存された設定と現在の設定が一致しない場合、`sync()` と `search()` は `ConfigMismatchError` を送出する。
ユーザーは `rebuild()` を実行する必要がある。

---

## 14. SQLite schema and migrations

SQLite database には schema version を保存する。

```sql
CREATE TABLE IF NOT EXISTS schema_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS collections (
    collection_name TEXT PRIMARY KEY,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    metadata_json TEXT
);

CREATE TABLE IF NOT EXISTS sources (
    collection_name TEXT NOT NULL,
    source_path TEXT NOT NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    PRIMARY KEY (collection_name, source_path),
    FOREIGN KEY (collection_name) REFERENCES collections(collection_name)
);

CREATE TABLE IF NOT EXISTS files (
    collection_name TEXT NOT NULL,
    file_path TEXT NOT NULL,
    source_path TEXT NOT NULL,
    relative_path TEXT NOT NULL,
    mtime REAL NOT NULL,
    size INTEGER NOT NULL,
    content_hash TEXT NOT NULL,
    last_indexed_at REAL NOT NULL,
    PRIMARY KEY (collection_name, file_path)
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY,
    collection_name TEXT NOT NULL,
    file_path TEXT NOT NULL,
    source_path TEXT NOT NULL,
    relative_path TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    token_count INTEGER,
    mtime REAL NOT NULL,
    created_at REAL NOT NULL
);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    chunk_id UNINDEXED,
    collection_name UNINDEXED,
    content
);
```

### 方針

- `schema_version` が現在のライブラリ version と互換でない場合、v1 では自動 migration を行わない
- 互換でない場合は明示的なエラーを送出する
- `chunks.content` と `chunks_fts.content` は duplicate storage でよい

---

## 15. Error classes

v1 では built-in exception でもよいが、公開 API として扱うなら以下の名前を用意する。

```python
class MdHybridSearchError(Exception):
    pass

class ConfigMismatchError(MdHybridSearchError):
    pass

class IndexCorruptionError(MdHybridSearchError):
    pass

class EmbeddingError(MdHybridSearchError):
    pass

class SourceNotFoundError(MdHybridSearchError):
    pass
```

最低限の実装としては、`ValueError`、`FileNotFoundError`、`RuntimeError` でもよい。

---

## 16. Non-goals in v1

v1 では以下を行わない。

- ファイル監視による自動 sync
- 複数プロセスからの同時 sync
- collection 横断検索
- AI 回答生成
- PDF / HTML / JSON / docx 等の読み込み
- Obsidian backlink / wikilink の解析
- frontmatter による filter
- tag filter
- heading-aware chunking
- chunk 単位の差分更新
- embedding cache
- automatic migration

---

## 17. 使い方の想定 (Obsidian Vault 向けの例)

Obsidian Vault を対象にした最小限の利用例。

```python
import os
from src.index import SearchIndex, DirectorySource

# 1. 呼び出し元で Embedder を用意 (例: OpenAI)
class OpenAIEmbedder:
    def __init__(self, api_key: str):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key)
        self.model_name = "text-embedding-3-small"
        self.embedding_dim = 1536

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        res = self.client.embeddings.create(input=texts, model=self.model_name)
        return [data.embedding for data in res.data]

    def embed_query(self, text: str) -> list[float]:
        res = self.client.embeddings.create(input=[text], model=self.model_name)
        return res.data[0].embedding

# 2. SearchIndex の初期化
vault_path = os.path.expanduser("~/Documents/Obsidian/MyVault")
app_data_dir = os.path.expanduser("~/Library/Application Support/my-app")
os.makedirs(app_data_dir, exist_ok=True)

index = SearchIndex(
    collection_name="my-vault",
    sources=[DirectorySource(vault_path)],
    sqlite_path=os.path.join(app_data_dir, "search.sqlite"),
    chroma_path=os.path.join(app_data_dir, "chroma"),
    embedder=OpenAIEmbedder(api_key=os.environ["OPENAI_API_KEY"]),
    chunk_size=1000,
    chunk_overlap=100
)

# 3. インデックスの同期 (差分更新)
report = index.sync()
print(f"Scanned: {report.scanned_files}, New: {report.new_files}, Deleted: {report.deleted_files}")

# 4. 検索
hits = index.search("AI search optimization", mode="hybrid", limit=5)
for hit in hits:
    print(f"Score: {hit.score:.4f} | Path: {hit.metadata['relative_path']}")
    # hit.content にはチャンクの本文が含まれる
```

この設計で、外部アプリは

- `app_data_dir` の管理
- `collection_name` と `sources` の管理
- `embedder` (および API Key) の管理

を担当し、ライブラリは検索インデックスの整合性維持とハイブリッド検索アルゴリズムに集中する。

---

## 18. 採用する前提

- v1 は Markdown 専用
- v1 は recursive scan 固定
- v1 は collection 単位の sync/search に集中する
- cross-collection search は公開 API に入れない
- エラーハンドリングは最小限にする
- source path の不在を削除扱いにしない
- chunking と embedder の変更には `rebuild()` を使う

---

## 19. 既存実装との対応

これは現行実装の参照メモであり、仕様の優先順位は上の章が上である。

- 差分スキャン: 既存の `src/md-hybrid-search/loader/FileIndex.py` でファイルテーブル管理
- チャンク化・埋め込み投入: `src/md-hybrid-search/loader/custom_directory_loader.py` でディレクトリのファイルリストを取得し、`src/md-hybrid-search/loader/custom_file_loader.py` でファイルタイプ別にローダー呼び出し、`src/md-hybrid-search/loader/custom_text_loader.py` で Markdown をロードしている
- SQLite FTS5 によるチャンク保存: `src/md-hybrid-search/loader/CustomFileLoader.py` にある `ChromaFTSWrapper` で ChromaDB へのドキュメント追加と FTS5 同期を行っている
- FTS 管理: `src/md-hybrid-search/search/fts_index.py`、`src/md-hybrid-search/search/fts_sync.py`、`src/md-hybrid-search/search/mecab_tokenizer.py`
- ハイブリッド検索: `src/md-hybrid-search/search/hybrid_search.py`
- reranker: `src/md-hybrid-search/search/reranker.py`


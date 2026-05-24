# md-hybrid-search 実装 ToDo

## 1. 公開 API と入力バリデーション
- [x] `DirectorySource` を定義する
- [x] `Embedder` protocol を定義する
- [x] `SearchIndex` のコンストラクタ引数を確定する
- [x] `sync()` / `search()` の公開インターフェースを確定する
- [x] `rebuild()` / `clear()` の公開インターフェースを確定する
- [x] `SyncReport` / `SearchHit` の戻り値型を定義する
- [x] `collection_name` のバリデーションを実装する
- [x] `DirectorySource.path` の正規化を実装する
- [x] 同一 source の重複除去を実装する
- [x] 親子関係にある source を `ValueError` にする
- [x] `sync()` の `sources` 空指定を `ValueError` にする
- [x] `sync()` の存在しない source を `FileNotFoundError` にする
- [x] `search()` の `query` / `limit` / `mode` バリデーションを実装する

## 2. SQLite スキーマと永続化
- [x] SQLite 接続処理を実装する
- [x] `schema_meta` / `collections` / `sources` / `files` / `chunks` / `chunks_fts` を設計・実装する
- [x] collection metadata に `chunk_size` / `chunk_overlap` / `embedder_fingerprint` / `tokenizer_fingerprint` / `schema_version` を保存する
- [x] schema version の保存と互換性チェックを実装する
- [x] 自動 migration を行わない方針を実装する
- [x] collection / source / file / chunk の CRUD を実装する

## 3. Markdown 読み込み・Chunking
- [x] Markdown を UTF-8 で読み込む処理を実装する
- [x] Markdown をプレーンテキストとして固定長 chunk に分割する処理を実装する
- [x] `chunk_size` / `chunk_overlap` を反映する
- [x] YAML frontmatter を本文に含める
- [x] heading-aware chunking を行わない
- [x] `chunk_id` を `collection_name` / `file_path` / `chunk_index` / `content_hash` から決定的に生成する
- [x] chunk に必要なメタデータを定義する
- [X] FTS 用の正規化テキストと query 正規化を実装する

## 4. Embedding
- [ ] 呼び出し元から渡された embedder で document embedding を生成する
- [ ] query embedding を生成する
- [ ] embedding dimension の整合性をチェックする
- [ ] embedding 生成失敗時の例外伝播方針を実装する
- [ ] embedding cache を作らない方針を反映する

## 5. Sync 処理
- [ ] source 配下を再帰スキャンする処理を実装する
- [ ] 対象を `.md` のみに絞る
- [ ] `mtime` / `size` / `content_hash` で追加・変更・未変更を判定する
- [ ] 新規ファイルの chunk 化・embedding・SQLite 保存・FTS 保存・Chroma 保存を実装する
- [ ] 変更ファイルの既存 chunk 削除と再投入を実装する
- [ ] 削除ファイルの SQLite / FTS / Chroma からの削除を実装する
- [ ] collection から外れた source 配下の file / chunk を削除する
- [ ] SQLite 更新を transaction 内で行う
- [ ] `SyncReport` を返す

## 6. FTS5 / BM25
- [ ] FTS5 用テーブルとインデックスを作成する
- [ ] chunk の追加・更新・削除に合わせて FTS を同期する
- [ ] collection 内に限定した keyword 検索を実装する
- [ ] BM25 結果を `SearchHit.score` 用に順位ベースへ変換する
- [ ] `SearchHit.score` は「大きいほど上位」になるようにする

## 7. ChromaDB
- [ ] `PersistentClient` を作成する
- [ ] `collection_name` ごとに Chroma collection を作成・取得する
- [ ] `chunk_id` 単位で upsert する処理を実装する
- [ ] `chunk_id` 単位で delete する処理を実装する
- [ ] Chroma metadata を SQLite の chunks と揃える

## 8. Hybrid Search
- [ ] keyword search と similarity search を実装する
- [ ] hybrid search で両方の候補を取得する
- [ ] candidate count を `max(limit * 5, 50)` にする
- [ ] RRF で統合する
- [ ] RRF の `k` を 60 にする
- [ ] 同一 `chunk_id` の結果を 1 件にまとめる
- [ ] 上位 `limit` 件を `SearchHit` として返す
- [ ] `SearchHit.metadata` に必要な項目を含める

## 9. Rebuild / Clear / エラー処理
- [ ] `rebuild()` を実装する
- [ ] `clear()` を実装する
- [ ] 保存済み設定と現在設定の不一致を検出する
- [ ] `ConfigMismatchError` を送出する方針を実装する
- [ ] schema version 不一致時のエラー方針を実装する
- [ ] 不整合時に `rebuild()` を案内する

## 10. テスト
- [ ] `collection_name` のバリデーションテストを書く
- [ ] `DirectorySource.path` の正規化テストを書く
- [ ] 重複 source と親子 source のテストを書く
- [ ] SQLite schema 作成・更新のテストを書く
- [ ] `chunk_id` 生成と chunking のテストを書く
- [ ] ファイル追加・変更・削除検出のテストを書く
- [ ] `sources` 不在と `FileNotFoundError` のテストを書く
- [ ] FTS / BM25 検索のテストを書く
- [ ] ChromaDB upsert / delete のテストを書く
- [ ] hybrid search と RRF のテストを書く
- [ ] `rebuild()` / `clear()` / config mismatch のテストを書く
- [ ] `sync()` と `search()` の統合テストを書く

## 11. ドキュメント・利用例
- [ ] 最小構成の `SearchIndex` 生成例を書く
- [ ] `sync()` / `search()` / `rebuild()` / `clear()` の使用例を書く
- [ ] 呼び出し元アプリ側の責務を書く
- [ ] `collection_name` ルールと rebuild が必要な条件を書く
- [ ] Obsidian Vault を対象にした利用例を書く

## 12. v1 対象外として整理
- [ ] 設定ファイル形式の設計はライブラリ責務から外す
- [ ] API key 保存はライブラリ責務から外す

### 前提
- v1 は Markdown 専用
- v1 は recursive scan 固定
- v1 は collection 単位の sync/search に集中する
- chunking と embedder の変更には `rebuild()` を使う
- cross-collection search は公開 API に入れない

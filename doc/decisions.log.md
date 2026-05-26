# Decisions Log

## 2026-05-25
- embedding dimension は `embedding_dim` / `dim` を優先し、無い場合は最初の embedding で観測した次元を検証用に保持する
- embedding cache は持たず、毎回 embedder を呼ぶ
- Sync 処理において、SQLite のトランザクションを `with self.db.conn:` で一括管理するようにし、`db.py` 内の個別メソッドからはトランザクション管理を外した。
- ChromaDB への反映は SQLite のコミット後に行うが、差分抽出時に削除対象の ID を収集しておくことで効率化した。
- ChromaDB への反映をファイル単位で行うように変更。全ファイル分のチャンクをメモリに保持して最後に一括で upsert するのではなく、ファイルごとに upsert する。
- ChromaDB の `max_batch_size` を超えるチャンクを持つ巨大なファイルに対応するため、内部で自動的にバッチ分割して upsert/delete を行うようにした。
- ソース設定から外れたディレクトリ配下のファイルも、`sync()` 実行時に自動的に削除対象として集計・処理するようにした。
- ハイブリッド検索の統合ロジックとして RRF (k=60) を採用し、候補取得数を `max(limit * 5, 50)` に固定した。
- `SearchIndex.sync()` 内で、常にコレクションのメタデータを更新するようにし、`updated_at` タイムスタンプが同期のたびに更新されるようにした。

## 2026-05-24
- v0からrerankerを受け取る実装に変更
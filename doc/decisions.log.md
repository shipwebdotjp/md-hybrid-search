# Decisions Log

## 2026-05-25
- embedding dimension は `embedding_dim` / `dim` を優先し、無い場合は最初の embedding で観測した次元を検証用に保持する
- embedding cache は持たず、毎回 embedder を呼ぶ
- Sync 処理において、SQLite のトランザクションを `with self.db.conn:` で一括管理するようにし、`db.py` 内の個別メソッドからはトランザクション管理を外した。
- ChromaDB への反映は SQLite のコミット後に行うが、差分抽出時に削除対象の ID を収集しておくことで効率化した。
- ソース設定から外れたディレクトリ配下のファイルも、`sync()` 実行時に自動的に削除対象として集計・処理するようにした。

## 2026-05-24
- v0からrerankerを受け取る実装に変更
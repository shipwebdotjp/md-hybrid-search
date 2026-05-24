import sqlite3
import json
from pathlib import Path

from src.search.fts_sync import get_fts_sync_manager


def _load_env_settings():
    """Load simple KEY=VALUE pairs from a .env file at project root.

    Values are returned as strings. Lines starting with # or empty lines are ignored.
    Leading `export ` is supported and quotes around values are stripped.
    """
    env_file = BASE_DIR / ".env"
    env_settings = {}
    if not env_file.exists():
        return env_settings
    try:
        with env_file.open(encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("export "):
                    line = line[len("export "):].lstrip()
                if "=" not in line:
                    continue
                key, val = line.split("=", 1)
                key = key.strip()
                val = val.strip()
                if (val.startswith('"') and val.endswith('"')) or (
                    val.startswith("'") and val.endswith("'")
                ):
                    val = val[1:-1]
                env_settings[key] = val
    except Exception:
        # If anything goes wrong reading .env, ignore and return empty dict
        return {}
    return env_settings

# プロジェクトルートディレクトリを取得
BASE_DIR = Path(__file__).resolve().parent
DB_DIR = BASE_DIR / "db"
SETTINGS_DB = DB_DIR / "settings.sqlite3"


def load_settings():
    con = sqlite3.connect(str(SETTINGS_DB))
    cur = con.cursor()
    cur.execute(
        "CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value JSON)"
    )
    res = cur.execute("SELECT value FROM settings")
    settings = res.fetchone()
    if settings:
        settings = json.loads(settings[0]) if settings[0] else {}
    else:
        settings = {}
    con.close()
    # Load .env settings (strings) and merge with DB settings.
    # Priority: DB settings override .env (DB > .env)
    env_settings = _load_env_settings()
    merged = env_settings.copy()
    # Ensure settings from DB overwrite env values
    if isinstance(settings, dict):
        merged.update(settings)

    return merged


def save_settings(new_settings):
    global settings
    con = sqlite3.connect(str(SETTINGS_DB))
    cur = con.cursor()
    for collection in settings.get("collection", []):
        if get_collection(collection.get("name"), new_settings) is None:
            delete_collection(collection.get("name"), cur)
        else:
            for source in collection.get("source"):
                if source.get("type") == "directory":
                    if (
                        get_source(
                            source.get("value"), collection.get("name"), new_settings
                        )
                        is None
                    ):
                        delete_document_by_dir(
                            collection.get("name"), source.get("value"), cur
                        )
    cur.execute(
        "INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)",
        ("settings", json.dumps(new_settings)),
    )
    con.commit()
    con.close()
    settings = new_settings


def get_collection(collection_name, settings):
    for collection in settings.get("collection"):
        if collection.get("name") == collection_name:
            return collection
    return None


def delete_document_by_dir(collection_name, dir_path, cur=None):
    import main

    db = main.get_db(collection_name)
    collection = db.get(where={"file_directory": dir_path})
    if len(collection["documents"]) > 0:
        ids = collection["ids"]
        db.delete(ids)
        # Delete from FTS index
        fts_sync = get_fts_sync_manager()
        for doc_id in ids:
            fts_sync.sync_delete(doc_id)
        if cur:
            cur.execute(
                """
                DELETE FROM file
                WHERE collection_name = ? AND file_directory = ?
            """,
                (collection_name, dir_path),
            )


def get_source(source_path, collection_name, settings):
    collection = get_collection(collection_name, settings)
    if collection is None:
        return None
    for source in collection.get("source", []):
        if source.get("value") == source_path:
            return source
    return None


def delete_collection(collection_name, cur=None):
    import main

    # Delete collection from vector DB
    main.get_db(collection_name).delete_collection()
    # Delete from FTS index
    get_fts_sync_manager().sync_delete_collection(collection_name)
    # Delete collection from settings DB file table
    if cur:
        cur.execute(
            """
            DELETE FROM file
            WHERE collection_name = ?
        """,
            (collection_name,),
        )


# global settings
settings = load_settings()

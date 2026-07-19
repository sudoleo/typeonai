from app.services import publisher_config


class Snap:
    def __init__(self, data):
        self.data = data

    @property
    def exists(self):
        return self.data is not None

    def to_dict(self):
        return dict(self.data or {})


class Ref:
    def __init__(self, store, key):
        self.store = store
        self.key = key

    def get(self):
        return Snap(self.store.get(self.key))

    def set(self, data, merge=False):
        if merge and self.key in self.store:
            self.store[self.key].update(data)
        else:
            self.store[self.key] = dict(data)


class Collection:
    def __init__(self, store):
        self.store = store

    def document(self, key):
        return Ref(self.store, key)


class Db:
    def __init__(self):
        self.store = {}

    def collection(self, name):
        assert name == "app_config"
        return Collection(self.store)


def test_default_publisher_configuration_is_persisted_and_free_pinned():
    db = Db()

    config = publisher_config.get_config(db=db)

    assert config["enabled"] is True
    assert config["weekly_watch_enabled"] is True
    assert db.store["scheduled_consensus_publisher"]["topic_brief"] == (
        publisher_config.DEFAULT_TOPIC_BRIEF
    )
    assert publisher_config.public_config(config)["watch_model_tier"] == "free"
    assert publisher_config.public_config(config)["excluded_providers"] == ["deepseek"]


def test_saved_publisher_configuration_is_normalized():
    db = Db()
    data = {
        **publisher_config.DEFAULT_CONFIG,
        "enabled": False,
        "watch_weekday": "Friday",
        "watch_time": "14:30",
        "watch_timezone": "Europe/Berlin",
    }

    saved = publisher_config.save_config(data, updated_by="admin-1", db=db)

    assert saved["enabled"] is False
    assert saved["watch_weekday"] == "friday"
    assert db.store["scheduled_consensus_publisher"]["updated_by"] == "admin-1"


def test_legacy_default_topic_brief_migrates_to_ai_search_strategy():
    config = publisher_config.normalize_config({
        **publisher_config.DEFAULT_CONFIG,
        "topic_brief": publisher_config.LEGACY_DEFAULT_TOPIC_BRIEF,
    })

    assert config["topic_brief"] == publisher_config.DEFAULT_TOPIC_BRIEF
    assert "highly current" in config["topic_brief"]
    assert "AI topic" in config["topic_brief"]

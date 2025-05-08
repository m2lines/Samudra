from ocean_emulators.config_base import BaseConfig, TopLevelConfig

# Complex config example for testing


class ServerCredentials(BaseConfig):
    username: str
    password: str


class DatabaseServer(BaseConfig):
    host: str
    port: int
    credentials: ServerCredentials


class CacheServer(BaseConfig):
    host: str
    port: int
    ttl: int


class RedisConfig(BaseConfig):
    main: CacheServer
    backup: CacheServer | None = None


class ApiConfig(BaseConfig):
    url: str
    token: str


class MetricsCollector(BaseConfig):
    enabled: bool
    interval: int


class BackendConfig(BaseConfig):
    databases: list[DatabaseServer]
    cache: RedisConfig
    api: ApiConfig
    metrics: MetricsCollector


class FrontendConfig(BaseConfig):
    api_url: str
    debug: bool


class AppConfig(TopLevelConfig):
    name: str
    version: str
    backend: BackendConfig
    frontend: FrontendConfig
    maintenance_mode: bool = False


# Create a complex config for tests
config = AppConfig(
    name="ComplexApp",
    version="2.0.0",
    backend=BackendConfig(
        databases=[
            DatabaseServer(
                host="primary-db.example.com",
                port=5432,
                credentials=ServerCredentials(username="admin", password="secret1"),
            ),
            DatabaseServer(
                host="replica-db.example.com",
                port=5432,
                credentials=ServerCredentials(username="reader", password="secret2"),
            ),
        ],
        cache=RedisConfig(
            main=CacheServer(host="redis.example.com", port=6379, ttl=300),
            backup=CacheServer(host="redis-backup.example.com", port=6379, ttl=300),
        ),
        api=ApiConfig(url="https://api.example.com/v2", token="abc123xyz"),
        metrics=MetricsCollector(enabled=True, interval=60),
    ),
    frontend=FrontendConfig(api_url="/api", debug=True),
    maintenance_mode=False,
)


def test_bind__binds_nested_baseconfig_by_type():
    @config.bind()
    def collect_metrics(collector: MetricsCollector):
        assert isinstance(collector, MetricsCollector)
        assert collector.enabled is True
        assert collector.interval == 60

    collect_metrics()


def test_bind__picks_first_option_based_on_type():
    @config.bind()
    def get_cache_server(cache: CacheServer):
        assert isinstance(cache, CacheServer)
        assert cache.host == "redis.example.com"
        assert cache.port == 6379
        assert cache.ttl == 300

    get_cache_server()


def test_bind__matches_both_name_and_type():
    @config.bind()
    def get_backup_server(backup: CacheServer):
        assert isinstance(backup, CacheServer)
        assert backup.host == "redis-backup.example.com"
        assert backup.port == 6379
        assert backup.ttl == 300

    get_backup_server()


def test_bind__parses_collections_of_types():
    @config.bind()
    def get_dbs(dbs: list[DatabaseServer]):
        assert len(dbs) == 2
        assert isinstance(dbs[0], DatabaseServer)
        assert dbs[0].host == "primary-db.example.com"
        assert dbs[1].host == "replica-db.example.com"
        assert dbs[1].credentials.username == "reader"

    get_dbs()


def test_bind__accepts_user_specified_access_path_mappings(cache: CacheServer):
    @config.bind(cache="backend.cache.backup")
    def get_cache_server(cache: CacheServer):
        assert isinstance(cache, CacheServer)
        assert cache.host == "redis-backup.example.com"
        assert cache.port == 6379
        assert cache.ttl == 300

    get_cache_server()

from dask.distributed import Client

# TODO: Make this path absolute (using Pathlib for example).
WORKER_TEMPLATE_PATH = "worker-template.yaml"
ALLOWED_CLUSTER_TYPES = ["kubernetes", "local"]

# TODO: Improve this class (methods and properties)


class DaskScaler(object):

    client = None

    def __init__(self, cluster_type):
        if cluster_type not in ALLOWED_CLUSTER_TYPES:
            raise Exception("Can't choose this type of cluster for now. Choose one from: {}".format(
                ALLOWED_CLUSTER_TYPES))
        self.cluster_type = cluster_type
        self._cluster = self.get_cluster()
        self._client = Client(self._cluster)

    @staticmethod
    def _get_kubernetes_cluster(worker_template_path=WORKER_TEMPLATE_PATH):
        from dask_kubernetes import KubeCluster

        cluster = KubeCluster.from_yaml(worker_template_path)
        return Client(cluster)

    @staticmethod
    def _get_local_cluster():
        # TODO: Add more parameters and configurations.
        from distributed import LocalCluster
        return LocalCluster()

    def get_cluster(self):
        return getattr(self, "_get_" + self.cluster_type + "_cluster")

    @property
    def cluster(self):
        if self._cluster is None:
            self._cluster = self.get_cluster()
        return self._cluster

    @property
    def client(self):
        if self._client is None:
            self._client = Client(self._cluster)
        return self._client

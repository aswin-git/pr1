from prometheus_client import Histogram, Counter

REQUEST_COUNT = Counter('request_count', 'Total no predictions')

REQUEST_LATENCY = Histogram('request_latency', 'Latency' )
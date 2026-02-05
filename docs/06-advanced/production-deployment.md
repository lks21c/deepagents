# 프로덕션 배포 가이드

> Deep Agents를 프로덕션 환경에 배포할 때 고려해야 할 사항입니다.

## 개요

프로덕션 환경에서 Deep Agents를 운영하려면 보안, 확장성, 관찰성, 비용 최적화 등 다양한 측면을 고려해야 합니다. 이 가이드에서는 엔터프라이즈 수준의 배포를 위한 모범 사례를 다룹니다.

---

## 아키텍처 고려사항

### 배포 토폴로지

```
                    ┌─────────────────┐
                    │   Load Balancer │
                    └────────┬────────┘
                             │
           ┌─────────────────┼─────────────────┐
           │                 │                 │
    ┌──────▼──────┐   ┌──────▼──────┐   ┌──────▼──────┐
    │  Agent Pod  │   │  Agent Pod  │   │  Agent Pod  │
    │  Instance 1 │   │  Instance 2 │   │  Instance 3 │
    └──────┬──────┘   └──────┬──────┘   └──────┬──────┘
           │                 │                 │
           └─────────────────┼─────────────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
       ┌──────▼──────┐  ┌────▼────┐  ┌─────▼─────┐
       │ PostgreSQL  │  │  Redis  │  │ LLM APIs  │
       │ Checkpoint  │  │  Cache  │  │(Anthropic,│
       │   Store     │  │         │  │  OpenAI)  │
       └─────────────┘  └─────────┘  └───────────┘
```

### 구성 요소별 책임

| 구성 요소 | 책임 | 확장 전략 |
|----------|------|----------|
| Agent Pod | 에이전트 실행, 요청 처리 | 수평 확장 (HPA) |
| PostgreSQL | 체크포인트, 세션 상태 | 읽기 복제본 |
| Redis | 캐싱, 속도 제한 | 클러스터 모드 |
| LLM APIs | 추론 | API 제공자에 의존 |

---

## 체크포인터 설정

### PostgreSQL 체크포인터 (권장)

```python
from langgraph.checkpoint.postgres import PostgresSaver
from deepagents import create_deep_agent

# 연결 문자열 (환경 변수에서 로드)
import os
DATABASE_URL = os.environ["DATABASE_URL"]

# PostgreSQL 체크포인터 생성
checkpointer = PostgresSaver.from_conn_string(DATABASE_URL)

# 에이전트 생성
agent = create_deep_agent(
    checkpointer=checkpointer,
)
```

### 데이터베이스 스키마 마이그레이션

```sql
-- LangGraph 체크포인터 스키마
CREATE TABLE IF NOT EXISTS checkpoints (
    thread_id VARCHAR(255) NOT NULL,
    checkpoint_id VARCHAR(255) NOT NULL,
    parent_checkpoint_id VARCHAR(255),
    type VARCHAR(50),
    checkpoint JSONB NOT NULL,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    PRIMARY KEY (thread_id, checkpoint_id)
);

CREATE INDEX idx_checkpoints_thread ON checkpoints(thread_id);
CREATE INDEX idx_checkpoints_created ON checkpoints(created_at);

-- 오래된 체크포인트 정리를 위한 정책
-- 예: 30일 이상 된 체크포인트 삭제
CREATE OR REPLACE FUNCTION cleanup_old_checkpoints()
RETURNS void AS $$
BEGIN
    DELETE FROM checkpoints
    WHERE created_at < NOW() - INTERVAL '30 days';
END;
$$ LANGUAGE plpgsql;
```

### 연결 풀링

```python
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool

# 연결 풀 설정
engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=10,           # 기본 연결 수
    max_overflow=20,        # 최대 추가 연결
    pool_timeout=30,        # 연결 대기 타임아웃
    pool_recycle=1800,      # 연결 재활용 (30분)
    pool_pre_ping=True,     # 연결 상태 확인
)
```

---

## 캐싱 전략

### Redis 캐시 설정

```python
from langgraph.cache.redis import RedisCache
import redis

# Redis 연결
redis_client = redis.Redis(
    host=os.environ.get("REDIS_HOST", "localhost"),
    port=int(os.environ.get("REDIS_PORT", 6379)),
    password=os.environ.get("REDIS_PASSWORD"),
    ssl=True,  # 프로덕션에서는 SSL 권장
)

# LLM 응답 캐시
cache = RedisCache(
    redis_client=redis_client,
    ttl=3600,  # 1시간 TTL
)

agent = create_deep_agent(
    cache=cache,
)
```

### 캐시 키 전략

```python
import hashlib
import json

def compute_cache_key(request) -> str:
    """결정적 캐시 키 생성

    동일한 입력에 대해 항상 같은 키를 생성합니다.
    """
    key_data = {
        "model": request.model,
        "messages": [
            {"role": m.type, "content": str(m.content)}
            for m in request.messages
        ],
        "tools": sorted([t.name for t in request.tools]),
        "temperature": request.temperature,
    }
    key_str = json.dumps(key_data, sort_keys=True)
    return hashlib.sha256(key_str.encode()).hexdigest()
```

---

## 백엔드 선택

### 프로덕션 환경별 권장 백엔드

| 환경 | 백엔드 | 이유 |
|-----|-------|------|
| 웹 서버 | StateBackend | 요청별 격리, 무상태 |
| 배치 처리 | FilesystemBackend | 영구 저장, 대용량 파일 |
| 컨테이너화 | StateBackend + StoreBackend | 유연한 조합 |
| 샌드박스 필요 | DockerBackend | 격리된 실행 |

### 프로덕션 백엔드 설정 예시

```python
from deepagents import create_deep_agent
from deepagents.backends import StateBackend
from deepagents.backends.composite import CompositeBackend
from langgraph.store.postgres import PostgresStore

# 영구 저장소 (메모리, 설정 등)
store = PostgresStore.from_conn_string(DATABASE_URL)

# 복합 백엔드: 임시 파일 + 영구 저장
def create_backend(runtime):
    return CompositeBackend(
        default=StateBackend(runtime),  # 임시 파일
        routes={
            "/memories/": store,  # 영구 저장
            "/configs/": store,
        },
    )

agent = create_deep_agent(
    backend=create_backend,
    store=store,
)
```

---

## 보안

### API 키 관리

```python
import os
from cryptography.fernet import Fernet

class SecureConfig:
    """안전한 설정 관리"""

    def __init__(self):
        # 암호화 키 (환경 변수에서 로드)
        self._cipher = Fernet(os.environ["ENCRYPTION_KEY"].encode())

    def get_api_key(self, provider: str) -> str:
        """API 키 복호화"""
        encrypted = os.environ.get(f"{provider.upper()}_API_KEY_ENCRYPTED")
        if not encrypted:
            raise ValueError(f"{provider} API key not configured")
        return self._cipher.decrypt(encrypted.encode()).decode()

# 사용
config = SecureConfig()
anthropic_key = config.get_api_key("anthropic")
```

### Human-in-the-Loop 강제

```python
from langchain.agents.middleware import HumanInTheLoopMiddleware, InterruptOnConfig

# 프로덕션에서는 위험한 작업에 HITL 필수
agent = create_deep_agent(
    checkpointer=checkpointer,  # HITL에 필수
    interrupt_on={
        # 파일 수정은 항상 승인 필요
        "edit_file": True,
        "write_file": True,

        # 실행은 조건부 승인
        "execute": InterruptOnConfig(
            condition=lambda args: any(
                cmd in args.get("command", "")
                for cmd in ["rm", "delete", "drop", "truncate"]
            ),
            message="위험한 명령이 감지되었습니다. 실행하시겠습니까?",
        ),
    },
)
```

### 입력 검증

```python
import re
from langchain.agents.middleware.types import AgentMiddleware, ModelRequest

class InputValidationMiddleware(AgentMiddleware):
    """입력 검증 미들웨어"""

    # 차단할 패턴
    BLOCKED_PATTERNS = [
        r"ignore\s+previous\s+instructions",
        r"system\s*:\s*",
        r"<\|.*\|>",  # 특수 토큰 패턴
    ]

    def __init__(self):
        self._patterns = [re.compile(p, re.I) for p in self.BLOCKED_PATTERNS]

    def wrap_model_call(self, request, handler):
        # 최신 메시지 검증
        for message in request.messages:
            if message.type == "human":
                content = str(message.content)
                for pattern in self._patterns:
                    if pattern.search(content):
                        raise ValueError("잠재적으로 악의적인 입력이 감지되었습니다.")

        return handler(request)
```

---

## 관찰성 (Observability)

### LangSmith 통합

```python
import os

# LangSmith 설정
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.environ["LANGSMITH_API_KEY"]
os.environ["LANGCHAIN_PROJECT"] = "production-agents"

# 에이전트 생성 (자동으로 트레이싱 활성화)
agent = create_deep_agent(
    debug=False,  # 프로덕션에서는 False
)
```

### 커스텀 메트릭 미들웨어

```python
import time
from prometheus_client import Counter, Histogram

# Prometheus 메트릭
llm_requests = Counter(
    "llm_requests_total",
    "Total LLM API requests",
    ["model", "status"],
)
llm_latency = Histogram(
    "llm_request_duration_seconds",
    "LLM request latency",
    ["model"],
)
tool_calls = Counter(
    "tool_calls_total",
    "Total tool calls",
    ["tool_name", "status"],
)

class MetricsMiddleware(AgentMiddleware):
    """Prometheus 메트릭 수집 미들웨어"""

    def wrap_model_call(self, request, handler):
        model_name = getattr(request, "model_name", "unknown")
        start_time = time.time()

        try:
            response = handler(request)
            llm_requests.labels(model=model_name, status="success").inc()
            return response
        except Exception as e:
            llm_requests.labels(model=model_name, status="error").inc()
            raise
        finally:
            duration = time.time() - start_time
            llm_latency.labels(model=model_name).observe(duration)

    def wrap_tool_call(self, request, handler):
        tool_name = request.tool_call["name"]

        try:
            result = handler(request)
            tool_calls.labels(tool_name=tool_name, status="success").inc()
            return result
        except Exception as e:
            tool_calls.labels(tool_name=tool_name, status="error").inc()
            raise
```

### 구조화된 로깅

```python
import structlog
import json

# 구조화된 로거 설정
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
)

logger = structlog.get_logger()

class LoggingMiddleware(AgentMiddleware):
    """구조화된 로깅 미들웨어"""

    def wrap_model_call(self, request, handler):
        logger.info(
            "llm_request_start",
            message_count=len(request.messages),
            tool_count=len(request.tools),
        )

        response = handler(request)

        logger.info(
            "llm_request_complete",
            has_tool_calls=bool(response.tool_calls),
            response_length=len(str(response.message.content)),
        )

        return response
```

---

## 확장성

### 수평 확장 (Kubernetes)

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: deep-agents
spec:
  replicas: 3
  selector:
    matchLabels:
      app: deep-agents
  template:
    metadata:
      labels:
        app: deep-agents
    spec:
      containers:
      - name: agent
        image: your-registry/deep-agents:latest
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "2000m"
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: db-credentials
              key: url
        - name: ANTHROPIC_API_KEY
          valueFrom:
            secretKeyRef:
              name: llm-credentials
              key: anthropic-key
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: deep-agents-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: deep-agents
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

### 속도 제한

```python
import redis
import time

class RateLimiter:
    """토큰 버킷 속도 제한"""

    def __init__(
        self,
        redis_client: redis.Redis,
        key_prefix: str = "ratelimit",
        requests_per_minute: int = 60,
    ):
        self.redis = redis_client
        self.prefix = key_prefix
        self.rate = requests_per_minute

    def check(self, identifier: str) -> bool:
        """요청 허용 여부 확인"""
        key = f"{self.prefix}:{identifier}"
        now = time.time()
        minute_start = int(now // 60) * 60

        # 트랜잭션으로 원자적 처리
        pipe = self.redis.pipeline()
        pipe.zadd(key, {str(now): now})
        pipe.zremrangebyscore(key, 0, minute_start)
        pipe.zcard(key)
        pipe.expire(key, 120)
        results = pipe.execute()

        count = results[2]
        return count <= self.rate

    def wait_if_needed(self, identifier: str):
        """필요시 대기"""
        while not self.check(identifier):
            time.sleep(1)
```

---

## 비용 최적화

### 모델 선택 전략

```python
from langchain.agents.middleware.types import AgentMiddleware, ModelRequest

class ModelRoutingMiddleware(AgentMiddleware):
    """작업 복잡도에 따른 모델 라우팅

    간단한 작업은 저렴한 모델, 복잡한 작업은 고성능 모델 사용
    """

    def __init__(self):
        self.cheap_model = "anthropic:claude-3-haiku-20240307"
        self.standard_model = "anthropic:claude-sonnet-4-5-20250929"
        self.premium_model = "anthropic:claude-opus-4-5-20250929"

    def _estimate_complexity(self, request: ModelRequest) -> str:
        """작업 복잡도 추정"""
        total_tokens = sum(
            len(str(m.content).split())
            for m in request.messages
        )

        # 간단한 휴리스틱
        if total_tokens < 500 and len(request.tools) < 3:
            return "simple"
        elif total_tokens < 2000:
            return "standard"
        else:
            return "complex"

    def wrap_model_call(self, request, handler):
        complexity = self._estimate_complexity(request)

        model_map = {
            "simple": self.cheap_model,
            "standard": self.standard_model,
            "complex": self.premium_model,
        }

        # 모델 오버라이드 (실제 구현에서는 model 교체 로직 필요)
        # modified_request = request.override(model=model_map[complexity])

        return handler(request)
```

### 토큰 사용량 모니터링

```python
import tiktoken

class TokenMonitoringMiddleware(AgentMiddleware):
    """토큰 사용량 모니터링"""

    def __init__(self, alert_threshold: int = 100000):
        self._encoder = tiktoken.get_encoding("cl100k_base")
        self._session_tokens = 0
        self._alert_threshold = alert_threshold

    def _count_tokens(self, text: str) -> int:
        """토큰 수 계산"""
        return len(self._encoder.encode(text))

    def wrap_model_call(self, request, handler):
        # 입력 토큰 계산
        input_tokens = sum(
            self._count_tokens(str(m.content))
            for m in request.messages
        )

        response = handler(request)

        # 출력 토큰 계산
        output_tokens = self._count_tokens(str(response.message.content))

        self._session_tokens += input_tokens + output_tokens

        # 임계값 초과 경고
        if self._session_tokens > self._alert_threshold:
            logger.warning(
                "token_threshold_exceeded",
                session_tokens=self._session_tokens,
                threshold=self._alert_threshold,
            )

        return response
```

---

## 재해 복구

### 체크포인트 백업

```python
import subprocess
from datetime import datetime

def backup_checkpoints(database_url: str, backup_path: str):
    """체크포인트 테이블 백업"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{backup_path}/checkpoints_{timestamp}.sql"

    subprocess.run([
        "pg_dump",
        "--table=checkpoints",
        "--data-only",
        "--file", filename,
        database_url,
    ], check=True)

    return filename

def restore_checkpoints(database_url: str, backup_file: str):
    """체크포인트 복원"""
    subprocess.run([
        "psql",
        "--file", backup_file,
        database_url,
    ], check=True)
```

### 헬스 체크 엔드포인트

```python
from fastapi import FastAPI, HTTPException
from typing import Dict

app = FastAPI()

@app.get("/health")
async def health_check() -> Dict[str, str]:
    """기본 헬스 체크"""
    return {"status": "healthy"}

@app.get("/ready")
async def readiness_check() -> Dict[str, str]:
    """준비 상태 체크"""
    try:
        # 데이터베이스 연결 확인
        checkpointer.get_latest_checkpoint("healthcheck")

        # Redis 연결 확인
        redis_client.ping()

        return {"status": "ready"}
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))
```

---

## 배포 체크리스트

### 필수 항목

- [ ] PostgreSQL 체크포인터 설정
- [ ] API 키 암호화 및 시크릿 관리
- [ ] HITL 미들웨어 활성화 (위험한 작업)
- [ ] 입력 검증 미들웨어 적용
- [ ] 구조화된 로깅 설정
- [ ] 헬스/준비 엔드포인트 구현
- [ ] 속도 제한 적용
- [ ] SSL/TLS 암호화

### 권장 항목

- [ ] LangSmith 트레이싱 활성화
- [ ] Prometheus 메트릭 수집
- [ ] Redis 캐싱 설정
- [ ] 자동 확장 (HPA) 구성
- [ ] 체크포인트 백업 스케줄링
- [ ] 모델 라우팅 최적화
- [ ] 토큰 사용량 모니터링

### 문서화

- [ ] 운영 런북 작성
- [ ] 장애 대응 절차 문서화
- [ ] 에스컬레이션 경로 정의
- [ ] SLA/SLO 정의

---

## 관련 문서

- [아키텍처 개요](../01-architecture/overview.md)
- [백엔드 시스템](../01-architecture/backend-system.md)
- [Human-in-the-Loop 패턴](../04-patterns/human-in-the-loop.md)
- [create_deep_agent API](../05-api-reference/create-deep-agent-api.md)

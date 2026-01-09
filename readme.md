flowchart LR
  %% =========================================================
  %% Functional Service Flow (End-to-End)
  %% =========================================================

  %% -------------------------
  %% Lanes (Swimlanes)
  %% -------------------------
  subgraph L1[데이터 수집 / 엣지]
    DC1[검측 주행\n(자동 검측 열차)]
    DC2[상단/하단 카메라\n이미지/영상 수집]
    DC3[선로 이미지\n프레임/클립 생성]
    DC4[전신주·애자 이미지\n프레임/클립 생성]
  end

  subgraph L2[플랫폼 Core (API/Workflow/DB)]
    P1[수집 데이터 적재\n(메타데이터 태깅: 노선/구간/시간)]
    P2[이벤트 오케스트레이션\n(추론 요청 생성/큐잉)]
    P3[(중앙 DB)\n선로 상태/결함 이력]
    P4[(문서/규정 DB)\n선로 규범·매뉴얼]
    P5[이상 이벤트 생성\n(결함종류/위치/신뢰도/증거)]
    P6[워크플로우 상태 관리\n(신규→검토→확정→조치)]
    P7[피드백 데이터셋 저장\n(이미지+교정라벨+메타)]
  end

  subgraph L3[AI 추론 / LLM 서비스]
    AI1[비전 추론\n레일 결함 탐지 모델]
    AI2[비전 추론\n전신주·애자 결함 탐지 모델]
    AI3[의사결정 지원 LLM\n(근거/유사사례/설명)]
    AI4[문서 초안 생성 LLM\n(정비 보고서 템플릿)]
    AI5[RAG 검색\n(규정/매뉴얼/과거 이력)]
  end

  subgraph L4[엔지니어 UI]
    U1[GIS 모니터링 UI\n(지도/구간/알림)]
    U2[이상 이벤트 상세 UI\n(증거/점수/설명)]
    U3[판정/라벨 교정 UI\n(오탐/미탐/등급)]
    U4[문서 확인 UI\n(초안/근거/수정)]
    U5[챗봇 UI\n(질의응답/근거 안내)]
  end

  subgraph L5[MLOps 자동 재학습/배포]
    M1[재학습 트리거\n(주기/데이터량/성능저하)]
    M2[파인튜닝 학습 파이프라인\n(데이터 검증/학습)]
    M3[평가/검증\n(리그레션/기준충족)]
    M4[모델 레지스트리\n(버전 관리)]
    M5[배포/가중치 업데이트\n(롤링/롤백)]
    M6[운영 모니터링\n(성능/드리프트)]
  end

  %% -------------------------
  %% Main Flow A: 수집 → 추론 → 이벤트 생성 → 모니터링
  %% -------------------------
  DC1 --> DC2
  DC2 --> DC3
  DC2 --> DC4

  DC3 --> P1
  DC4 --> P1
  P1 --> P2

  P2 --> AI1
  P2 --> AI2

  AI1 --> P5
  AI2 --> P5

  P5 --> P3
  P5 --> P6
  P6 --> U1
  U1 --> U2

  %% -------------------------
  %% Main Flow B: 엔지니어 판독/확정 + 의사결정 지원
  %% -------------------------
  U2 --> AI3
  AI3 --> U2
  U2 --> U3
  U3 --> P6
  P6 --> P3

  %% -------------------------
  %% Main Flow C: 정비 보고서 초안 생성 (RAG + LLM)
  %% -------------------------
  U4 --> AI5
  AI5 --> P4
  P3 --> AI5
  AI5 --> AI4
  AI4 --> U4
  U4 --> P6

  %% -------------------------
  %% Main Flow D: 피드백 데이터 축적 (라벨/샘플 저장)
  %% -------------------------
  U3 --> P7
  P7 --> P3

  %% -------------------------
  %% Main Flow E: 자동 재학습 → 평가 → 배포 → 운영 모니터링
  %% -------------------------
  P7 --> M1
  M1 --> M2
  M2 --> M3
  M3 -->|통과| M4
  M4 --> M5
  M5 --> M6
  M3 -->|미달| M2
  M6 -->|성능저하 감지| M1

  %% -------------------------
  %% Optional: 챗봇(규정/이력 질의)
  %% -------------------------
  U5 --> AI5
  AI5 --> U5

  %% -------------------------
  %% Styling
  %% -------------------------
  classDef edge fill:#FFF7E6,stroke:#F4B183,stroke-width:1px,color:#333;
  classDef platform fill:#E8F4FF,stroke:#5B9BD5,stroke-width:1px,color:#333;
  classDef ai fill:#E9F7EF,stroke:#70AD47,stroke-width:1px,color:#333;
  classDef ui fill:#F2F2F2,stroke:#7F7F7F,stroke-width:1px,color:#333;
  classDef mlops fill:#FDECEF,stroke:#C00000,stroke-width:1px,color:#333;
  classDef db fill:#FFFFFF,stroke:#333,stroke-width:1.2px,color:#333;

  class DC1,DC2,DC3,DC4 edge;
  class P1,P2,P5,P6,P7 platform;
  class AI1,AI2,AI3,AI4,AI5 ai;
  class U1,U2,U3,U4,U5 ui;
  class M1,M2,M3,M4,M5,M6 mlops;
  class P3,P4 db;
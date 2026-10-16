# dsz-bus-2026
2026 데이터안심구역 경진대회 출품작

9401 광역버스 탑승가능성 기반 경로 안내 서비스

## 데이터 (전부 데이터안심구역 데이터)
- 교통카드 이용내역(수도권), 노선 정보, 노선정류장 정보, 정류장 정보 (한국교통안전공단)
- SKT 시간대별·요일별 유동인구


## 구성
| 파일 | 내용 |
|---|---|
| data-engineering/01RouteStops.py | 9401 노선ID, 정류장 순서·좌표·서울 여부 |
| data-engineering/02CardExtract.py | 교통카드에서 9401 기록 + 정류장별 전체 노선 승차 |
| data-engineering/03BusEvents.py | 버스 한 대씩 도착 시각·재차·앞차와의 공백 |
| data-engineering/04LoadProfile.py | 정류장x시간대 하루 평균(평일/주말/요일), 9401 분담률 |
| data-engineering/05SktCatchment.py | 정류장 반경 SKT 유동인구 |
| data-engineering/06BuildFeatures.py | 합치기 (서울 도심 구간) |
| modeling/07BoardingProbability.py | 수요 복원 + 탑승가능확률 |
| modeling/08Validate.py | 교차검증, 포화 곡선, 민감도, 몰림 효과 |
| modeling/09ServiceRecommend.py | 기대 대기 시간, 정류장·위치 기반 추천(서울을 빠져나가기까지 가장 빠른 정류장), 대기 부담 |
| modeling/10ExportResults.py | 반출용 표 (원자료 컬럼 자동 차단) |
| modeling/Figures.py | 반출용 그림 22개 (탐색 E1~E5, 결과·검증·서비스 F0~F16) |

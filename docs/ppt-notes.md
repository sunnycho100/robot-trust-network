# PPT 메모

## Slide 1. 지금까지의 커뮤니케이션 방식: ROS2 Node Subscription

**슬라이드 내용**
- 로봇 = 역할별로 나뉜 작은 프로그램(Node)들의 모음
  - 예: 카메라 Node, 경로 계획 Node, 모터 제어 Node
- Publish / Subscribe 구조
  - Publisher가 Topic(예: `/cmd_vel`)에 메시지를 보냄
  - 그 Topic을 Subscribe한 Node가 메시지를 받음
- 메시지 타입이 정해져 있음 (예: 속도 명령 = 직선 속도 + 회전 속도)
- 내부적으로 DDS가 같은 네트워크 안의 Node를 자동으로 찾아서 연결

**스크립트**
> 지금까지 로봇은 주로 ROS2의 Node Subscription 방식으로 통신해 왔습니다. 로봇 안에는 카메라, 경로 계획, 모터 제어처럼 역할별로 나뉜 Node들이 있고, 각 Node는 Topic이라는 채널에 메시지를 올리거나 구독합니다. 보내는 쪽은 누가 받는지 몰라도 되기 때문에 부품을 추가하거나 바꾸기가 쉽습니다. 다만 이 구조를 쓰려면 ROS2 환경을 직접 세팅하고, Topic 이름과 메시지 형식을 알고 있어야 합니다.

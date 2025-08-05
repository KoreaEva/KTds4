# Azure AI Search 솔루션 만들기
모든 조직은 정보에 의존하여 의사 결정을 내리고, 질문에 답하고, 효율적으로 기능합니다. 대부분의 조직에서 문제는 정보 부족이 아니라 정보가 저장된 방대한 문서, 데이터베이스 및 기타 소스에서 정보를 찾고 추출하는 문제입니다.

예를 들어 Margie's Travel이 전 세계 도시로의 여행을 조직하는 것을 전문으로 하는 여행사라고 가정해 보겠습니다. 시간이 지남에 따라 회사는 브로셔와 같은 문서와 고객이 제출한 호텔 리뷰에 방대한 양의 정보를 축적했습니다. 이 데이터는 여행사 직원과 고객이 여행을 계획할 때 유용한 인사이트 소스이지만, 데이터의 양이 너무 많아 특정 고객 질문에 답할 수 있는 관련 정보를 찾기가 어려울 수 있습니다.

이 문제를 해결하기 위해 Margie's Travel은 Azure AI Search를 사용하여 AI 기술을 사용하여 문서를 더 쉽게 검색할 수 있도록 문서를 인덱싱하고 보강하는 솔루션을 구현할 수 있습니다.
Azure 리소스 만들기
Margie's Travel에 대해 만들 솔루션에는 Azure 구독에 다음 리소스가 필요합니다.

- 인덱싱 및 쿼리를 관리하는 Azure AI Search 리소스입니다.
- Azure AI Services 리소스는 검색 솔루션이 AI에서 생성된 인사이트로 데이터 원본의 데이터를 보강하는 데 사용할 수 있는 기술에 대한 AI 서비스를 제공합니다.
- 검색할 문서가 저장되는 Blob 컨테이너가 있는 스토리지 계정입니다.
중요: Azure AI Search 및 Azure AI Services 리소스는 동일한 위치에 있어야 합니다.

## Azure AI Search 리소스 만들기
1.	웹 브라우저에서 에서 Azure Portal을 열고 Azure 구독과 연결된 Microsoft 계정을 사용하여 로그인합니다.
https://portal.azure.com
2.	+리소스 만들기 단추를 선택하고, 검색을 검색하고, 다음 설정을 사용하여 Azure AI Search 리소스를 만듭니다.
- 구독: Azure 구독
- 리소스 그룹: 새 리소스 그룹 만들기(제한된 구독을 사용하는 경우 새 리소스 그룹을 만들 수 있는 권한이 없을 수 있습니다. 제공된 리소스 그룹 사용)
- 서비스 이름: 고유한 이름을 입력합니다.
- 위치: 위치 선택 - Azure AI Search 및 Azure AI Services 리소스는 동일한 위치에 있어야 합니다.
- 가격 책정 계층: 기본
3.	배포가 완료될 때까지 기다린 다음, 배포된 리소스로 이동합니다.
4.	Azure Portal에서 Azure AI Search 리소스에 대한 블레이드의 개요 페이지를 검토합니다. 여기에서 시각적 인터페이스를 사용하여 검색 솔루션의 다양한 구성 요소를 생성, 테스트, 관리 및 모니터링할 수 있습니다. 데이터 원본, 인덱스, 인덱서 및 기술 세트를 포함합니다.

## Azure AI Services 리소스 만들기
 
구독에 아직 없는 경우 Azure AI Services 리소스를 프로비전해야 합니다. 검색 솔루션은 이를 사용하여 AI가 생성한 인사이트로 데이터 저장소의 데이터를 보강합니다.
1.	Azure Portal의 홈페이지로 돌아가서 +리소스 만들기 단추를 선택하고, Azure AI Services를 검색하고, 다음 설정을 사용하여 Azure AI Services 다중 서비스 계정 리소스를 만듭니다.

- 구독: Azure 구독
- 리소스 그룹: Azure AI Search 리소스와 동일한 리소스 그룹입니다
- 지역: Azure AI Search 리소스와 동일한 위치
- 이름: 고유한 이름을 입력합니다.
- 가격 책정 계층: 표준 S0
 
2.	필요한 확인란을 선택하고 리소스를 만듭니다.
3.	배포가 완료될 때까지 기다린 다음 배포 세부 정보를 확인합니다.

## 스토리지 계정 만들기
 
1.	Azure Portal의 홈페이지로 돌아가서 +리소스 만들기 단추를 선택하고, 스토리지 계정을 검색하고, 다음 설정을 사용하여 스토리지 계정 리소스를 만듭니다.
- 구독: Azure 구독
- 리소스 그룹: *Azure AI Search 및 Azure AI Services 리소스와 동일한 리소스 그룹
- 스토리지 계정 이름: 고유한 이름을 입력합니다.
- 지역: 사용 가능한 지역을 선택합니다.
- 성능 : 표준
- 중복성: LRS(로컬 중복 스토리지)
- Advanced(고급) 탭에서 Allow enabling anonymous access on individual containers(개별 컨테이너에 대한 익명 액세스 허용 옆의 확인란을 선택합니다)
2.	배포가 완료될 때까지 기다린 다음, 배포된 리소스로 이동합니다.
3.	개요 페이지에서 구독 ID - 스토리지 계정이 프로비전된 구독을 식별합니다.
 
4.	액세스 키 페이지에서 스토리지 계정에 대해 두 개의 키가 생성되었습니다. 그런 다음 키 표시를 선택하여 키를 봅니다.
 
> 팁: 저장소 계정 블레이드를 열어 둡니다. 다음 절차에서 구독 ID와 키 중 하나가 필요합니다.
Visual Studio Code에서 앱 개발 준비
Visual Studio Code를 사용하여 검색 앱을 개발합니다. 앱의 코드 파일은 GitHub 리포지토리에 제공되었습니다.

> 팁: mslearn-knowledge-mining 리포지토리를 이미 복제한 경우 Visual Studio 코드에서 엽니다. 그렇지 않으면 다음 단계에 따라 개발 환경에 복제합니다.

## Azure AI Search Index 생성 준비

1.	Visual Studio Code를 시작합니다.
2.	팔레트를 열고(Shift+CTRL+P) Git: Clone 명령을 실행하여 리포지토리를 로컬 폴더에 복제합니다(어떤 폴더인지는 상관없음).
https://github.com/MicrosoftLearning/mslearn-knowledge-mining
3.	리포지토리가 복제되면 Visual Studio Code에서 폴더를 엽니다.
4.	리포지토리에서 C# 코드 프로젝트를 지원하기 위해 추가 파일이 설치되는 동안 기다립니다.
참고: 빌드 및 디버그에 필요한 자산을 추가하라는 메시지가 표시되면 지금 아님을 선택합니다.
Azure Storage에 문서 업로드
이제 필요한 리소스가 있으므로 Azure Storage 계정에 일부 문서를 업로드할 수 있습니다.
1.	Visual Studio Code의 탐색기 창에서 Labfiles\01-azure-search 폴더를 확장하고 UploadDocs.cmd 선택합니다.
2.	배치 파일을 편집하여 YOUR_SUBSCRIPTION_ID, YOUR_AZURE_STORAGE_ACCOUNT_NAME 및 YOUR_AZURE_STORAGE_KEY 자리 표시자를 이전에 만든 스토리지 계정에 대한 적절한 구독 ID, Azure Storage 계정 이름 및 Azure Storage 계정 키 값으로 바꿉니다.
3.	변경 내용을 저장한 다음 01-azure-search 폴더를 마우스 오른쪽 단추로 클릭하고 통합 터미널을 엽니다.
4.	Azure CLI를 사용하여 Azure 구독에 로그인하려면 다음 명령을 입력합니다.


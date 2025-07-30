# AI Skillset을 이용한 인덱스 생성

> 팁: 이 작업을 하기 전에 먼저 인덱스를 삭제해야 한다. 

Explore and modify definitions of search components(검색 구성 요소의 정의)
검색 솔루션의 구성 요소는 Azure Portal에서 보고 편집할 수 있는 JSON 정의를 기반으로 합니다.
포털을 사용하여 검색 솔루션을 만들고 수정할 수 있지만 JSON에서 검색 개체를 정의하고 Azure AI Service REST 인터페이스를 사용하여 만들고 수정하는 것이 바람직한 경우가 많습니다.\

## Azure AI Search 리소스에 대한 엔드포인트 및 키 가져오기
1.	Azure Portal에서 Azure AI Search 리소스에 대한 개요 페이지로 돌아갑니다. 그리고 페이지의 맨 위 섹션에서 리소스의 URL(예: https://resource_name.search.windows.net)을 찾아 클립보드에 복사합니다.

2.	Visual Studio Code의 탐색기 창에서 01-azure-search 폴더와 modify-search 하위 폴더를 확장하고 modify-search.cmd 선택하여 엽니다. 이 스크립트 파일을 사용하여 JSON을 Azure AI 서비스 REST 인터페이스에 제출하는 cURL 명령을 실행합니다.

3.	modify-search.cmd에서 YOUR_SEARCH_URL 자리 표시자를 클립보드에 복사한 URL로 바꿉니다.

4.	Azure Portal의 설정 섹션에서 Azure AI Search 리소스에 대한 키 페이지를 보고 기본 관리자 키를 클립보드에 복사합니다.

5.	Visual Studio Code에서 YOUR_ADMIN_KEY 자리 표시자를 클립보드에 복사한 키로 바꿉니다.

6.	변경 사항을 modify-search.cmd에 저장합니다(아직 실행하지는 마세요!)

## 기술 세트 검토 및 수정

1.	Visual Studio Code의 modify-search 폴더에서 skillset.json를 엽니다. 이는 margies-skillset에 대한 JSON 정의를 보여줍니다.

2.	기술 세트 정의의 맨 위에서 Azure AI Services 리소스를 기술 세트에 연결하는 데 사용되는 cognitiveServices 개체를 확인합니다.

3.	Azure Portal에서 Azure AI Services 리소스(Azure AI Search 리소스가 아님)를 열고 리소스 관리 섹션에서 키 및 엔드포인트 페이지를 확인합니다. 그런 다음 KEY 1을 클립보드에 복사합니다.

4.	Visual Studio Code의 skillset.json에서 YOUR_COGNITIVE_SERVICES_KEY 자리 표시자를 클립보드에 복사한 Azure AI Services 키로 바꿉니다.

5.	JSON 파일을 스크롤하여 Azure Portal에서 Azure AI Search 사용자 인터페이스를 사용하여 만든 기술에 대한 정의가 포함되어 있는지 확인합니다. 기술 목록의 맨 아래에 다음 정의와 함께 추가 기술이 추가되었습니다.

```json
{
    "@odata.type": "#Microsoft.Skills.Text.V3.SentimentSkill",
    "defaultLanguageCode": "en",
    "name": "get-sentiment",
    "description": "New skill to evaluate sentiment",
    "context": "/document",
    "inputs": [
        {
            "name": "text",
            "source": "/document/merged_content"
        },
        {
            "name": "languageCode",
            "source": "/document/language"
        }
    ],
    "outputs": [
        {
            "name": "sentiment",
            "targetName": "sentimentLabel"
        }
    ]
}
```

새 기술의 이름은 get-sentiment이며, 문서의 각 문서 수준에 대해 인덱싱되는 문서의 merged_content 필드에 있는 텍스트(원본 콘텐츠와 콘텐츠의 이미지에서 추출한 텍스트 포함)를 평가합니다. 문서에서 추출된 언어(기본값은 영어)를 사용하고 콘텐츠의 감정에 대한 레이블을 평가합니다. 감정 레이블의 값은 "긍정적", "부정적", "중립" 또는 "혼합"일 수 있습니다. 그런 다음 이 레이블은 sentimentLabel이라는 새 필드로 출력됩니다.

6.	skillset.json에 대한 변경 사항을 저장합니다.

## 인덱스 검토 및 수정
1.	Visual Studio Code의 modify-search 폴더에서 index.json 엽니다. 다음은 margies-index에 대한 JSON 정의를 보여줍니다.

2.	인덱스를 스크롤하여 필드 정의를 확인합니다. 일부 필드는 원본 문서의 메타데이터 및 콘텐츠를 기반으로 하고 다른 필드는 기술 세트의 기술 결과입니다.

3.	Azure Portal에서 정의한 필드 목록의 끝에 두 개의 필드가 추가되었습니다.

```json
{
    "name": "sentiment",
    "type": "Edm.String",
    "facetable": false,
    "filterable": true,
    "retrievable": true,
    "sortable": true
},
{
    "name": "url",
    "type": "Edm.String",
    "facetable": false,
    "filterable": true,
    "retrievable": true,
    "searchable": false,
    "sortable": false
}
```

4.	sentiment 필드는 기술 세트에 추가된 get-sentiment 기술의 출력을 추가하는 데 사용됩니다. url 필드는 데이터 소스에서 추출된 metadata_storage_path 값을 기반으로 인덱싱된 각 문서의 URL을 인덱스에 추가하는 데 사용됩니다. index에는 이미 metadata_storage_path 필드가 포함되어 있지만 인덱스 키와 Base-64로 인코딩되어 키로 효율적이지만 실제 URL 값을 필드로 사용하려면 클라이언트 응용 프로그램이 디코딩해야 합니다. 인코딩되지 않은 값에 대한 두 번째 필드를 추가하면 이 문제가 해결됩니다.

## 인덱서 검토 및 수정

1.	Visual Studio Code의 modify-search 폴더에서 indexer.json 엽니다. 이는 문서 콘텐츠 및 메타데이터에서 추출한 필드(fieldMappings 섹션)와 기술 세트의 기술에서 추출한 값(outputFieldMappings 섹션)을 인덱스의 필드에 매핑하는 margies-indexer에 대한 JSON 정의를 보여 줍니다.

2.	fieldMappings 목록에서 base-64로 인코딩된 키 필드에 대한 metadata_storage_path 값의 매핑을 확인합니다. metadata_storage_path 키로 할당하고 Azure Portal에서 키를 인코딩하는 옵션을 선택했을 때 생성되었습니다. 또한 새 매핑은 동일한 값을 url 필드에 명시적으로 매핑하지만 Base-64 인코딩은 사용하지 않습니다.

```json
{
    "sourceFieldName" : "metadata_storage_path",
    "targetFieldName" : "url"
}  
```

소스 문서의 다른 모든 메타데이터 및 콘텐츠 필드는 인덱스에서 동일한 이름의 필드에 암시적으로 매핑됩니다.

3.	기술 세트에 있는 기술의 출력을 인덱스 필드에 매핑하는 outputFieldMappings 섹션을 검토합니다. 이들 중 대부분은 사용자 인터페이스에서 선택한 사항을 반영하지만, 감정 기술에서 추출한 sentimentLabel 값을 인덱스에 추가한 감정 필드에 매핑하기 위해 다음 매핑이 추가되었습니다.

```json
{
    "sourceFieldName": "/document/sentimentLabel",
    "targetFieldName": "sentiment"
}
```

## REST API를 사용하여 검색 솔루션 업데이트

1.	modify-search 폴더를 마우스 오른쪽 버튼으로 클릭하고 통합 터미널을 엽니다.

2.	modify-search 폴더의 터미널 창에서 다음 명령을 입력하여 JSON 정의를 REST 인터페이스에 제출하고 인덱싱을 시작하는 modify-search.cmd 스크립트를 실행합니다.

```powershell
./modify-search
```
3.	스크립트가 완료되면 Azure Portal에서 Azure AI Search 리소스의 개요 페이지로 돌아가서 인덱서 페이지를 확인합니다. 주기적으로 Refresh(새로 고침)를 선택하여 인덱싱 작업의 진행 상황을 추적합니다. 완료하는 데 1분 정도 걸릴 수 있습니다.
감정을 평가하기에 너무 큰 몇 가지 문서에 대한 몇 가지 경고가 있을 수 있습니다. 감정 분석은 전체 문서가 아닌 페이지 또는 문장 수준에서 수행되는 경우가 많습니다. 그러나 이 경우 대부분의 문서, 특히 호텔 리뷰는 유용한 문서 수준 감정 점수를 평가할 수 있을 만큼 짧습니다.

## 수정된 인덱스 쿼리

1. Azure AI Search 리소스에 대한 블레이드 맨 위에서 검색 탐색기를 선택합니다.

2. 검색 탐색기의 쿼리 문자열 상자에서 다음 JSON 쿼리를 제출합니다.

```json
{
  "search": "London",
  "select": "url,sentiment,keyphrases",
  "filter": "metadata_author eq 'Reviewer' and sentiment eq 'positive'"
}

이 쿼리는 검토자가 작성한 런던을 언급하고 긍정적인 감정 레이블이 있는 모든 문서(즉, 런던을 언급하는 긍정적인 리뷰)에 대한 url, 감정 및 핵심 문구를 검색합니다

3. Search explorer(검색 탐색기) 페이지를 닫고 Overview(개요) 페이지로 돌아갑니다.
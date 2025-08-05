# 인덱스 검색

이제 인덱스가 있으므로 인덱스를 검색할 수 있습니다.

1. Azure AI Search 리소스에 대한 개요 페이지 맨 위에서 검색 탐색기를 선택합니다.

2. Search 탐색기의 Query string(쿼리 문자열) 상자에 (단일 별표)를 입력한 다음, Search(검색)를 선택합니다. * 이 쿼리는 인덱스의 모든 문서를 JSON 형식으로 검색합니다. 결과를 검사하고 선택한 인식 기술로 추출한 문서 콘텐츠, 메타데이터 및 보강된 데이터가 포함된 각 문서의 필드를 확인합니다.

3. 보기 메뉴에서 JSON 보기를 선택하고 검색에 대한 JSON 요청이 다음과 같이 표시되는지 확인합니다.
```json
{
  "search": "*"
}
```

4. 다음과 같이 count 매개 변수를 포함하도록 JSON 요청을 수정합니다.
```json
{
  "search": "*",
  "count": true
}
```

5. 수정된 검색을 제출합니다. 이번에는 검색에서 반환된 문서 수를 나타내는 결과 맨 위에 @odata.count 필드가 결과에 포함됩니다.

6. 다음 쿼리를 시도해 보세요.
```json
{
  "search": "*",
  "count": true,
  "select": "metadata_storage_name,metadata_author,locations"
}
```

이번에는 파일 이름, 작성자 및 문서 컨텐츠에 언급된 모든 위치만 결과에 포함됩니다. 파일 이름과 작성자는 소스 문서에서 추출된 metadata_storage_name 및 metadata_author 필드에 있습니다. 위치 필드는 인지 능력에 의해 생성되었습니다.

7. 이제 다음 쿼리 문자열을 사용해 보세요.
```json
{
  "search": "New York",
  "count": true,
  "select": "metadata_storage_name,keyphrases"
}
```
이 검색은 검색 가능한 필드에서 "New York"을 언급하는 문서를 찾고 문서에서 파일 이름과 핵심 구를 반환합니다.

8. 쿼리를 하나 더 시도해 보겠습니다.
```json
{
  "search": "New York",
  "count": true,
  "select": "metadata_storage_name",
  "filter": "metadata_author eq 'Reviewer'"
}
```

이 쿼리는 "New York"을 언급하는 검토자가 작성한 모든 문서의 파일 이름을 반환합니다.
# Azure CLI를 설치하는 방법

## Azure CLI 설치
Azure CLI의 현재 버전은 2.74.0입니다. 최신 릴리스에 대한 자세한 내용은 릴리스 정보를 참조하세요. 설치된 버전을 찾고 업데이트해야 하는지 확인하려면 az 버전을 실행합니다.

### Windows에 설치

- [WinGet(Windows 패키지 관리자)](https://learn.microsoft.com/ko-kr/cli/azure/install-azure-cli-windows?view=azure-cli-latest&pivots=winget) 사용하여 Windows에 설치
- [MSI(Microsoft Installer) 사용하여](https://learn.microsoft.com/ko-kr/cli/azure/install-azure-cli-windows?view=azure-cli-latest&pivots=msi) Windows에 설치
- [PowerShell MSI(Microsoft Installer)를 사용하여](https://learn.microsoft.com/ko-kr/cli/azure/install-azure-cli-windows?view=azure-cli-latest&pivots=msi-powershell) Windows에 설치
- [ZIP 패키지 사용하여](https://learn.microsoft.com/ko-kr/cli/azure/install-azure-cli-windows?view=azure-cli-latest&pivots=zip) Windows에 설치

### Linux 또는 WSL(Linux용 Windows 하위 시스템)에 설치(WSL이란?)

- [dnf를 사용하여](https://learn.microsoft.com/ko-kr/cli/azure/install-azure-cli-linux?view=azure-cli-latest&pivots=dnf) RHEL/CentOS Stream에 설치
- [zypper를 사용하여](https://learn.microsoft.com/ko-kr/cli/azure/install-azure-cli-linux?view=azure-cli-latest&pivots=zypper) SLES/OpenSUSE에 설치
- [apt를 사용하여](https://learn.microsoft.com/ko-kr/cli/azure/install-azure-cli-linux?view=azure-cli-latest&pivots=apt) Ubuntu/Debian에 설치
- [tdnf를 사용하여](https://learn.microsoft.com/ko-kr/cli/azure/install-azure-cli-linux?view=azure-cli-latest&pivots=tdnf) Azure Linux에 설치
- [스크립트에서](https://learn.microsoft.com/ko-kr/cli/azure/install-azure-cli-linux?view=azure-cli-latest&pivots=script) 설치

### macOS에 설치
- [macOS 버전으로](https://learn.microsoft.com/ko-kr/cli/azure/install-azure-cli-macos?view=azure-cli-latest) 설치

### Docker Container에 설치
- [Docker 컨테이너에서](https://learn.microsoft.com/ko-kr/cli/azure/run-azure-cli-docker?view=azure-cli-latest) 실행

### Azure Cloud Shell의 활용
- [Azure Cloud Shell](https://learn.microsoft.com/ko-kr/azure/cloud-shell/quickstart)의 활용

## 어떤 버전의 Azure CLI가 설치되어 있나요?
> 터미널 창에 입력 az version 하여 설치된 Azure CLI 버전을 확인합니다. 출력은 다음과 같습니다.

- 출력

```json
{
  "azure-cli": "x.xx.0x",
  "azure-cli-core": "x.xx.x",
  "azure-cli-telemetry": "x.x.x",
  "extensions": {}
}
```
어떤 확장이 설치되어 있나요?
az extension list 명령을 사용하여 설치된 확장을 확인합니다. 사용할 수도 az version있지만 az extension list 설치 경로 및 상태를 비롯한 추가 정보를 제공합니다. 확장 관리에 대한 자세한 내용은 Azure CLI를 사용하여 확장 사용 및 관리를 참조하세요.
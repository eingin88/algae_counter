# 藻類自動判讀與計數工具
內容使用大量AI編譯，請謹慎使用。  
  
使用方法  
a. 資料夾設置  
	請先建立一個資料夾，需包含以下內容  
	資料夾格式data、outputs  
    data內部放置顯微鏡影像  
	  
b. 專案設置  
	設置專案可以使安裝的插件隨資料夾刪除時一併移除  
    請在搜尋打開cmd(命令提示字元)  
    設定檔案位置輸入cd C:\xxx \xxx (您設置的資料夾路徑)  
	輸入 python -m venv venv  
	(如果無顯示東西，請確認python 是否安裝)  
	輸入venv\Scripts\activate  
    前面多了(venv)即為成功，資料夾會多出venv的文件夾  
	  
c. 插件下載  
	請先設置一個文字文件requirements.txt，放置於同一文件夾中並輸入以下插件。  
	numpy  
	pandas  
	opencv-python  
	scikit-image  
	tqdm  
	matplotlib  
	cellpose  
	完成後在cmd輸入pip install -r requirements.txt  
	如果不知道有沒有安裝pip，可以先確認版本pip –version  
	如果沒有可以輸入，python -m ensurepip –upgrade  
	  

# Nvidia GPU的情況下，要下載Torch Cuda以啟用GPU設定。
如果僅有CPU，需修改程式碼呼叫CellposeModel(gpu=false)，或直接刪除後綴。# 不建議使用CPU跑程式。  
網址: https://pytorch.org/get-started/locally/  
網址引導處會下載torch CUDA，請先檢查Nvidia內建的CUDA版本下載。(可以在命令提示字元輸入nvidia-smi檢查)  

d.影像比例校正  
如果您使用其他影像編輯軟體處理，只需確認解析度沒有放大或縮小即可。  
    以PPT疊圖為例  
	1.請先在PPT以寬23,高10cm的畫布以四張疊圖，並檢查圖片像素。  
    2.如果圖片尺寸像素低於下圖的尺寸，請依據比例將畫布放大，反之則縮小。  
    3. 使用measure_px直接量測帶有比例尺的四張疊圖，量測到的像素與micrometer的比例即為需要設定的比例。(measure_px需要更改路徑找圖)  
	  
e. 分析影像  
    請先以記事本打開run主程式，需要修改影像的路徑，只需一處。  
    在記事本搜尋: BASE_DIR = Path，並將後面路徑改成正確路徑  
	把要分析的影像放到 `data/images/`  
	啟動 venv 後執行：run.py  
	Cmd會顯示進度與count結果。  
	也可以到 `outputs/` 取結果與圖片報告：  
    

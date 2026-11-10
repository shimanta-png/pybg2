# Bulk Bg Remover 
This is an tool which uses to remove background from images. 

## Requirements
* Make sure your system should have latest python installed.
* windows -> keep visual c++ download from microsoft official repo[https://aka.ms/vc14/vc_redist.x64.exe?utm_source=chatgpt.com]


### Activation
For installing py packages use `python -m venv env`
then use, `source env\Scripts\activate`

This tool use a small ai model in use so after activating virtual enviroment, use this `python -c "import onnxruntime as ort; print(ort.__version__); print(ort.get_available_providers())"`
it will returns... 
```
1.31.1
['AzureExecutionProvider', 'CPUExecutionProvider']
```


### Testing
```
    python bulk_bg_remover.py input_folder output_folder
    python bulk_bg_remover.py input_folder output_folder -f same
    python bulk_bg_remover.py input_folder output_folder -f jpg
    python bulk_bg_remover.py input_folder output_folder -f jpg --bg-color "#ffffff"
    python bulk_bg_remover.py input_folder output_folder -f webp -r
    python bulk_bg_remover.py input_folder output_folder -f jpg --compress=80
    python bulk_bg_remover.py input_folder output_folder -f webp --compress=60
```
=> install the models: python -m scripts.download_models 

=> list connecte devices: v4l2-ctl --list-devices 
v4l2-ctl means `video for linux 2 control`


=> capture the photos: venv/bin/python3 -m scripts.capture_photos --name derick --camera 2

=> Command to enroll: venv/bin/python -m scripts.enroll

=> Command to identify: venv/bin/python -m scripts.recognize --camera 2

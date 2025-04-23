## Update by Xingrui

- Focal length in projection is 1400
`/home/xingrui/superclevr-physics/OmniNeMoSuperClever/nemo/models/project_kp.py `, line 91

'''
        cameras = PerspectiveCameras(focal_length=raster_configs.get('focal_length', 3000) / raster_configs.get('down_rate'), principal_point=((feature_size[1] // 2, feature_size[0] // 2,), ), image_size=(feature_size, ), in_ndc=False, device=device)
'''


- Change the Superclevr Dataset
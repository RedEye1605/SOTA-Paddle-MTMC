-- Sample cameras — matches configs/cameras.yaml.
-- The actual RTSP URL comes from the env var named in `rtsp_url_env_key`.

INSERT INTO cameras (camera_id, name, rtsp_url_env_key, site_id, timezone,
                     width, height, fps_target, is_active)
VALUES
    ('CAM_01', 'Entrance Main',     'CAM_01_RTSP_URL', 'showroom_jakarta_pusat', 'Asia/Jakarta', 1920, 1080, 25, TRUE),
    ('CAM_02', 'Showroom Floor N',   'CAM_02_RTSP_URL', 'showroom_jakarta_pusat', 'Asia/Jakarta', 1920, 1080, 25, TRUE),
    ('CAM_03', 'Showroom Floor S',   'CAM_03_RTSP_URL', 'showroom_jakarta_pusat', 'Asia/Jakarta', 1920, 1080, 25, TRUE),
    ('CAM_04', 'VIP Lounge',         'CAM_04_RTSP_URL', 'showroom_jakarta_pusat', 'Asia/Jakarta', 1280,  720, 15, TRUE),
    ('CAM_05', 'Workshop',           'CAM_05_RTSP_URL', 'showroom_jakarta_pusat', 'Asia/Jakarta', 1920, 1080, 15, TRUE)
ON CONFLICT (camera_id) DO NOTHING;

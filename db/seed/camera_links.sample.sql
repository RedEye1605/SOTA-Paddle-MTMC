-- Sample camera_links — matches configs/camera_links.yaml.

INSERT INTO camera_links (from_camera_id, to_camera_id, min_travel_seconds,
                          max_travel_seconds, transition_probability, enabled, notes)
VALUES
    ('CAM_01', 'CAM_02', 10,  90,  0.85, TRUE,  'After entering, customers walk to the showroom floor'),
    ('CAM_01', 'CAM_03', 15, 120,  0.70, TRUE,  'Some customers go directly to the bike display'),
    ('CAM_02', 'CAM_03', 20, 180,  0.60, TRUE,  'Walking between floor areas'),
    ('CAM_02', 'CAM_04', 30, 300,  0.40, TRUE,  'VIP escalation, requires staff escort'),
    ('CAM_03', 'CAM_05', 30, 180,  0.30, TRUE,  'Workshop drop-off'),
    -- explicit impossibilities (gating enforced even if ReID says match)
    ('CAM_01', 'CAM_04',  0,   0,  0.00, FALSE, 'Cannot reach VIP lounge without passing CAM_02 first'),
    ('CAM_04', 'CAM_01',  0,   0,  0.00, FALSE, 'VIP lounge has no direct exit to entrance')
ON CONFLICT (from_camera_id, to_camera_id) DO NOTHING;

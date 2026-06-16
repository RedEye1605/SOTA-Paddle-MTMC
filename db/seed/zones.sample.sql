-- Sample zones — matches configs/zones.yaml.

INSERT INTO zones (zone_id, camera_id, name, polygon_json, zone_type,
                   is_entry_zone, is_exit_zone, enabled)
VALUES
    ('Z01_CAM01_ENTRY',   'CAM_01', 'Main Entrance',
     '[[0.40, 0.00], [0.60, 0.00], [0.60, 0.50], [0.40, 0.50]]',
     'entry', TRUE, FALSE, TRUE),
    ('Z02_CAM01_EXIT',    'CAM_01', 'Main Exit',
     '[[0.40, 0.50], [0.60, 0.50], [0.60, 1.00], [0.40, 1.00]]',
     'exit', FALSE, TRUE, TRUE),
    ('Z03_CAM02_FLOOR',   'CAM_02', 'Showroom Floor North',
     '[[0.10, 0.10], [0.90, 0.10], [0.90, 0.90], [0.10, 0.90]]',
     'floor', FALSE, FALSE, TRUE),
    ('Z04_CAM03_DISPLAY', 'CAM_03', 'Bike Display Area',
     '[[0.20, 0.30], [0.80, 0.30], [0.80, 0.80], [0.20, 0.80]]',
     'display', FALSE, FALSE, TRUE),
    ('Z05_CAM04_VIP',     'CAM_04', 'VIP Lounge',
     '[[0.15, 0.15], [0.85, 0.15], [0.85, 0.85], [0.15, 0.85]]',
     'vip', FALSE, FALSE, TRUE),
    ('Z06_CAM05_WORKSHOP','CAM_05', 'Workshop Bay',
     '[[0.10, 0.10], [0.90, 0.10], [0.90, 0.90], [0.10, 0.90]]',
     'workshop', FALSE, FALSE, TRUE)
ON CONFLICT (zone_id) DO NOTHING;

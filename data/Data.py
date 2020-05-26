import os
from google_drive_downloader import GoogleDriveDownloader as gdd


class Downloader(object):
    def __init__(self, id=8, main_dir='dataset'):
        self.sequence_id = id
        if not os.path.exists(main_dir):
            os.mkdir(main_dir)
        if not os.path.exists(os.path.join(main_dir, 'sequences')):
            os.mkdir(os.path.join(main_dir, 'sequences'))

    def download_sequence(self):
        sequence = Sequence(self.sequence_id)
        gdd.download_file_from_google_drive(file_id=sequence.calib.id, dest_path=sequence.calib.name, unzip=True)
        gdd.download_file_from_google_drive(file_id=sequence.poses.id, dest_path=sequence.poses.id, unzip=True)
        gdd.download_file_from_google_drive(file_id=sequence.images.id, dest_path=sequence.images.id, unzip=True)


class Sequence(object):
    def __init__(self):
        self.calib = Kitti_link('data_odometry_calib.zip', '1jW1Yr8qBD2m63QQjN_q_EJWiQIyhtFj0')
        self.images = Kitti_link('data_odometry_color.zip', '1s6GhV8UQHdZjWaX1pcJy_8TZ9rbT-21C')
        self.poses = Kitti_link('data_odometry_poses.zip', '1m1J7T_1hvrIWbT14m9KDSrffgqhUaEfL')


class Kitti_link(object):
    def __init__(self, name, id):
        self.name = name
        self.id = id

if __name__=='__main__':
    print(Downloader(8))
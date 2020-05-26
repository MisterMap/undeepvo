import os

from google_drive_downloader import GoogleDriveDownloader as gdd


class Downloader(object):
    def __init__(self, sequence_id=8, main_dir='dataset'):
        self.sequence = Sequence(sequence_id)
        if not os.path.exists(main_dir):
            os.mkdir(main_dir)

    def download_sequence(self):
        gdd.download_file_from_google_drive(file_id=self.sequence.calib.id, dest_path=self.sequence.calib.name,
                                            unzip=True)
        gdd.download_file_from_google_drive(file_id=self.sequence.poses.id, dest_path=self.sequence.poses.name,
                                            unzip=True)
        gdd.download_file_from_google_drive(file_id=self.sequence.images.id, dest_path=self.sequence.images.name,
                                            unzip=True)
        self.clean_space()

    def clean_space(self):
        os.remove(self.sequence.calib.name)
        os.remove(self.sequence.poses.name)
        os.remove(self.sequence.images.name)


class Sequence(object):
    def __init__(self, sequence_id=8):
        self.sequence_id = sequence_id
        self.calib = Kitti_link('data_odometry_calib.zip', '1jW1Yr8qBD2m63QQjN_q_EJWiQIyhtFj0')
        self.images = Kitti_link('data_odometry_color.zip', '1s6GhV8UQHdZjWaX1pcJy_8TZ9rbT-21C')
        self.poses = Kitti_link('data_odometry_poses.zip', '1m1J7T_1hvrIWbT14m9KDSrffgqhUaEfL')


class Kitti_link(object):
    def __init__(self, name, id):
        self.name = os.path.join(os.curdir, name)
        self.id = id


if __name__ == '__main__':
    s8 = Downloader(8)
    s8.download_sequence()

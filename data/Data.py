import os
import shutil

from google_drive_downloader import GoogleDriveDownloader as gdd


class Downloader(object):
    def __init__(self, sequence_id='08', main_dir='dataset'):
        self.sequence_id = sequence_id
        self.main_dir = main_dir
        self.sequence = Sequence(self.sequence_id)
        if not os.path.exists(self.main_dir):
            os.mkdir(self.main_dir)

    def download_sequence(self):
        gdd.download_file_from_google_drive(file_id=self.sequence.calib.id, dest_path=self.sequence.calib.name,
                                            unzip=True)
        gdd.download_file_from_google_drive(file_id=self.sequence.poses.id, dest_path=self.sequence.poses.name,
                                            unzip=True)
        gdd.download_file_from_google_drive(file_id=self.sequence.images.id, dest_path=self.sequence.images.name,
                                            unzip=True)
        self.clean_space()

    def clean_space(self):
        shutil.move(os.path.join(os.curdir, self.sequence_id, 'image2'), os.path.join(os.curdir, self.main_dir, 'sequences', self.sequence_id, 'image2'))
        shutil.move(os.path.join(os.curdir, self.sequence_id, 'image2'), os.path.join(os.curdir, self.main_dir, 'sequences', self.sequence_id, 'image2'))
        os.remove(self.sequence.calib.name)
        os.remove(self.sequence.poses.name)
        os.remove(self.sequence.images.name)
        os.remove(os.path.join(os.curdir, self.sequence_id, 'calib.txt'))
        os.remove(os.path.join(os.curdir, self.sequence_id, 'times.txt'))
        os.remove(os.path.join(os.curdir, self.sequence_id))


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

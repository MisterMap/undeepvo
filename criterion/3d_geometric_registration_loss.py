import torch


class ThreeDGeometricRegistrationLoss(torch.nn.Module):
    def __init__(self, transformation_matrix):
        """

        :param transformation_matrix: transformation matrix from previous frame to the new one
        """
        # Transformation maatrix from previou
        super().__init__()
        self.transformation_matrix = transformation_matrix

    def generate_next_pcd(self, previous_pcd):
        return self.transformation_matrix @ previous_pcd

    def generate_previous_pcd(self, next_pcd):
        inverse_matrix = torch.inverse(self.transformation_matrix)
        return inverse_matrix @ next_pcd

    def forward(self, point_cloud_previous, point_cloud_next):
        generated_pcd_next = self.generate_next_pcd(point_cloud_previous)
        generated_pcd_previos = self.generate_previos_pcd(point_cloud_next)

        l1_loss = torch.nn.L1Loss()

        loss_previous = l1_loss(generated_pcd_previos, point_cloud_previous)
        loss_next = l1_loss(generated_pcd_next, point_cloud_next)

        return loss_previous + loss_next
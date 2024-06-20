import torch

class data_process:
    def __init__(self, X, Y=None, normal_y_mode=0):
        # Compute mean and standard deviation for X
        self.X_mean = X.mean(dim=0, keepdim=True)
        self.X_std = X.std(dim=0, keepdim=True) + 1e-6  # Avoid division by zero

        # Compute mean and standard deviation for Y if provided
        if Y is not None:
            if normal_y_mode == 0:
                self.Y_mean = Y.mean()
                self.Y_std = Y.std() + 1e-6  # Avoid division by zero
            elif normal_y_mode == 1:
                # option 2: normalize y by each dimension
                self.Ymean = Y.mean(0)
                self.Ystd = Y.std(0)
                self.Y = (Y - self.Ymean.expand_as(Y)) / (self.Ystd.expand_as(Y) + EPS)
            else:
                self.Y_mean = torch.zeros(1)
                self.Y_std = torch.ones(1)
        else:
            self.Y_mean = torch.zeros(1)
            self.Y_std = torch.ones(1)

    def normalize(self, X, Y=None):
        # Normalize X
        X_normalized = (X - self.X_mean) / self.X_std
        # Normalize Y if provided
        if Y is not None:
            Y_normalized = (Y - self.Y_mean) / self.Y_std
            return X_normalized, Y_normalized
        return X_normalized

    def denormalize(self, X, Y=None):
        # Denormalize X
        X_denormalized = X * self.X_std + self.X_mean
        # Denormalize Y if provided
        if Y is not None:
            Y_denormalized = Y * self.Y_std + self.Y_mean
            return X_denormalized, Y_denormalized
        return X_denormalized
    def denormalize_result(self, mean, var=None):
        # Denormalize the mean and variance of the prediction
        mean_denormalized = mean * self.Y_std + self.Y_mean
        if var is not None:
            var_denormalized = var * (self.Y_std ** 2)
            return mean_denormalized, var_denormalized
        return mean_denormalized
    def remove(self, X, Y, threshold=1e-4):
        # Remove points that are too close to each other, which can cause numerical instability and matrix singularity
        # Calculate pairwise distances
        distances = torch.cdist(X, X, p=2)

        # Create a mask to identify points within the threshold
        mask = distances < threshold

        # Zero out the diagonal to ignore self-comparison
        mask.fill_diagonal_(False)

        # Use the upper triangle of the mask to avoid double marking
        mask_upper = torch.triu(mask, diagonal=1)

        # Identify indices of points to keep
        to_remove = mask_upper.any(dim=0)
        to_keep = ~to_remove

        return X[to_keep], Y[to_keep]
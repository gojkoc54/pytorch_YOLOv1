from torchvision import transforms
import torchvision
import torch
import torch.nn as nn
from util_layers import Squeeze, Flatten, Print



class Darknet(nn.Module):

    def __init__(self, init_weights = True, fast_version = False, num_classes = 1000):
        super(Darknet, self).__init__()

        self.conv_layers = self.create_conv_layers(fast_version = fast_version)

        self.num_classes = num_classes
        self.classification_layers = self.create_classification_layers()

        if (init_weights == True):
            self.initialize_weights()



    def forward(self, x):

        out = self.conv_layers(x)

        out = self.classification_layers(out)

        return out



    def create_classification_layers(self):

        classification_layers = torch.nn.Sequential(

            nn.AvgPool2d(kernel_size = (14, 14)),
            Squeeze(),
            nn.Linear(1024, self.num_classes),
            nn.Softmax(dim = 1)
            
            # Dim = 1 because the first dimension will be the batch-size dimension !!!
            # Either way, watch out for this !!!

        )

        return classification_layers



    def create_conv_layers(self, fast_version = False):

        if (fast_version == False):

            conv_layers = torch.nn.Sequential(

                nn.Conv2d(3, 64, kernel_size = (7, 7), stride = 2, padding = 3),
                nn.BatchNorm2d(64, track_running_stats = True, affine=True),
                nn.LeakyReLU(0.1, inplace = True),
                nn.MaxPool2d(kernel_size = (2, 2), stride = 2),

                # ==============================================

                nn.Conv2d(64, 192, kernel_size = (3, 3), padding = 1),
                nn.BatchNorm2d(192, track_running_stats = True, affine=True),
                nn.LeakyReLU(0.1, inplace = True),
                nn.MaxPool2d(kernel_size = (2, 2), stride = 2),

                # ==============================================

                nn.Conv2d(192, 128, kernel_size = (1, 1)),
                nn.BatchNorm2d(128, track_running_stats = True, affine=True),
                nn.LeakyReLU(0.1, inplace = True),

                nn.Conv2d(128, 256, kernel_size = (3, 3), padding = 1),
                nn.BatchNorm2d(256, track_running_stats = True, affine=True),
                nn.LeakyReLU(0.1, inplace = True),

                nn.Conv2d(256, 256, kernel_size = (1, 1)),
                nn.BatchNorm2d(256, track_running_stats = True, affine=True),
                nn.LeakyReLU(0.1, inplace = True),

                nn.Conv2d(256, 512, kernel_size = (3, 3), padding = 1),
                nn.BatchNorm2d(512, track_running_stats = True, affine=True),
                nn.LeakyReLU(0.1, inplace = True),

                nn.MaxPool2d(kernel_size = (2, 2), stride = 2),

                # ==============================================
                
                nn.Conv2d(512, 256, kernel_size = (1, 1)),
                nn.BatchNorm2d(256, track_running_stats = True, affine=True),
                nn.LeakyReLU(0.1, inplace = True),

                nn.Conv2d(256, 512, kernel_size = (3, 3), padding = 1),
                nn.BatchNorm2d(512, track_running_stats = True, affine=True),
                nn.LeakyReLU(0.1, inplace = True),

                
                nn.Conv2d(512, 256, kernel_size = (1, 1)),
                nn.BatchNorm2d(256, track_running_stats = True, affine=True),
                nn.LeakyReLU(0.1, inplace = True),

                nn.Conv2d(256, 512, kernel_size = (3, 3), padding = 1),
                nn.BatchNorm2d(512, track_running_stats = True, affine=True),
                nn.LeakyReLU(0.1, inplace = True),

                
                nn.Conv2d(512, 256, kernel_size = (1, 1)),
                nn.BatchNorm2d(256, track_running_stats = True, affine=True),
                nn.LeakyReLU(0.1, inplace = True),

                nn.Conv2d(256, 512, kernel_size = (3, 3), padding = 1),
                nn.BatchNorm2d(512, track_running_stats = True, affine=True),
                nn.LeakyReLU(0.1, inplace = True),

                
                nn.Conv2d(512, 256, kernel_size = (1, 1)),
                nn.BatchNorm2d(256, track_running_stats = True, affine=True),
                nn.LeakyReLU(0.1, inplace = True),

                nn.Conv2d(256, 512, kernel_size = (3, 3), padding = 1),
                nn.BatchNorm2d(512, track_running_stats = True, affine=True),
                nn.LeakyReLU(0.1, inplace = True),

                
                nn.Conv2d(512, 512, kernel_size = (1, 1)),
                nn.BatchNorm2d(512, track_running_stats = True, affine=True),
                nn.LeakyReLU(0.1, inplace = True),
                
                nn.Conv2d(512, 1024, kernel_size = (3, 3), padding = 1),
                nn.BatchNorm2d(1024, track_running_stats = True, affine=True),
                nn.LeakyReLU(0.1, inplace = True),
                
                nn.MaxPool2d(kernel_size = (2, 2), stride = 2),

                # ==============================================

                nn.Conv2d(1024, 512, kernel_size = (1, 1)),
                nn.BatchNorm2d(512, track_running_stats = True, affine=True),
                nn.LeakyReLU(0.1, inplace = True),

                nn.Conv2d(512, 1024, kernel_size = (3, 3), padding = 1),
                nn.BatchNorm2d(1024, track_running_stats = True, affine=True),
                nn.LeakyReLU(0.1, inplace = True),


                nn.Conv2d(1024, 512, kernel_size = (1, 1)),
                nn.BatchNorm2d(512, track_running_stats = True, affine=True),
                nn.LeakyReLU(0.1, inplace = True),

                nn.Conv2d(512, 1024, kernel_size = (3, 3), padding = 1),
                nn.BatchNorm2d(1024, track_running_stats = True, affine=True),
                nn.LeakyReLU(0.1, inplace = True),

            )


        if (fast_version == True):

            conv_layers = torch.nn.Sequential(

                nn.Conv2d(3, 64, kernel_size = (7, 7), stride = 2, padding = 3),
                nn.BatchNorm2d(64, track_running_stats = True, affine=True),
                nn.LeakyReLU(0.1, inplace = True),
                nn.MaxPool2d(kernel_size = (2, 2), stride = 2),

                # ==============================================

                nn.Conv2d(64, 192, kernel_size = (3, 3), padding = 1),
                nn.BatchNorm2d(192, track_running_stats = True, affine=True),
                nn.LeakyReLU(0.1, inplace = True),
                nn.MaxPool2d(kernel_size = (2, 2), stride = 2),

                # ==============================================

                nn.Conv2d(192, 128, kernel_size = (1, 1)),
                nn.BatchNorm2d(128, track_running_stats = True, affine=True),
                nn.LeakyReLU(0.1, inplace = True),

                nn.Conv2d(128, 256, kernel_size = (3, 3), padding = 1),
                nn.BatchNorm2d(256, track_running_stats = True, affine=True),
                nn.LeakyReLU(0.1, inplace = True),

                nn.Conv2d(256, 512, kernel_size = (3, 3), padding = 1),
                nn.BatchNorm2d(512, track_running_stats = True, affine=True),
                nn.LeakyReLU(0.1, inplace = True),

                nn.MaxPool2d(kernel_size = (2, 2), stride = 2),

                # ==============================================
                
                nn.Conv2d(512, 256, kernel_size = (1, 1)),
                nn.BatchNorm2d(256, track_running_stats = True, affine=True),
                nn.LeakyReLU(0.1, inplace = True),

                nn.Conv2d(256, 512, kernel_size = (3, 3), padding = 1),
                nn.BatchNorm2d(512, track_running_stats = True, affine=True),
                nn.LeakyReLU(0.1, inplace = True),
                
                nn.Conv2d(512, 512, kernel_size = (1, 1)),
                nn.BatchNorm2d(512, track_running_stats = True, affine=True),
                nn.LeakyReLU(0.1, inplace = True),
                
                nn.Conv2d(512, 1024, kernel_size = (3, 3), padding = 1),
                nn.BatchNorm2d(1024, track_running_stats = True, affine=True),
                nn.LeakyReLU(0.1, inplace = True),
                
                nn.MaxPool2d(kernel_size = (2, 2), stride = 2),

                # ==============================================

                nn.Conv2d(1024, 512, kernel_size = (1, 1)),
                nn.BatchNorm2d(512, track_running_stats = True, affine=True),
                nn.LeakyReLU(0.1, inplace = True),

                nn.Conv2d(512, 1024, kernel_size = (3, 3), padding = 1),
                nn.BatchNorm2d(1024, track_running_stats = True, affine=True),
                nn.LeakyReLU(0.1, inplace = True),

            )

        return conv_layers



    def initialize_weights(self):
        
        for module in self.modules():
            
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='leaky_relu')
                
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.constant_(module.bias, 0)


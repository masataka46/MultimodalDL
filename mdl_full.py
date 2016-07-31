import chainer
import chainer.functions as F
import chainer.links as L


class MDL_full(chainer.Chain):

    """Single-GPU model of stage3 written in 'Multimodal Deep Learning for Robust RGB-D Object Recognition' without partition toward the channel axis."""

    insize = 227

    def __init__(self):
        super(MDL_full, self).__init__(
            convR1=L.Convolution2D(3,  96, 11, stride=4),
            convR2=L.Convolution2D(96, 256,  5, pad=2),
            convR3=L.Convolution2D(256, 384,  3, pad=1),
            convR4=L.Convolution2D(384, 384,  3, pad=1),
            convR5=L.Convolution2D(384, 256,  3, pad=1),
            fcR6=L.Linear(9216, 4096),
            fcR7=L.Linear(4096, 4096),
            convD1=L.Convolution2D(3,  96, 11, stride=4),
            convD2=L.Convolution2D(96, 256,  5, pad=2),
            convD3=L.Convolution2D(256, 384,  3, pad=1),
            convD4=L.Convolution2D(384, 384,  3, pad=1),
            convD5=L.Convolution2D(384, 256,  3, pad=1),
            fcD6=L.Linear(9216, 4096),
            fcD7=L.Linear(4096, 4096),
            fc8=L.Bilinear(4096, 4096, 4096),
            fc9=L.Linear(4096, 1000),
        )
        self.train = True

    def clear(self):
        self.loss = None
        self.accuracy = None

    def __call__(self, x, y, t):
        self.clear()
        hR = F.max_pooling_2d(F.relu(
            F.local_response_normalization(self.convR1(x))), 3, stride=2)
        hR = F.max_pooling_2d(F.relu(
            F.local_response_normalization(self.convR2(hR))), 3, stride=2)
        hR = F.relu(self.convR3(hR))
        hR = F.relu(self.convR4(hR))
        hR = F.max_pooling_2d(F.relu(self.convR5(hR)), 3, stride=2)
        hR = F.dropout(F.relu(self.fcR6(hR)), train=self.train)
        hR = F.dropout(F.relu(self.fcR7(hR)), train=self.train)
        hD = F.max_pooling_2d(F.relu(
            F.local_response_normalization(self.convD1(y))), 3, stride=2)
        hD = F.max_pooling_2d(F.relu(
            F.local_response_normalization(self.convD2(hD))), 3, stride=2)
        hD = F.relu(self.convD3(hD))
        hD = F.relu(self.convD4(hD))
        hD = F.max_pooling_2d(F.relu(self.convD5(hD)), 3, stride=2)
        hD = F.dropout(F.relu(self.fcD6(hD)), train=self.train)
        hD = F.dropout(F.relu(self.fcD7(hD)), train=self.train)
        h = F.dropout(F.relu(self.fc8(hR, hD)), train=self.train)
        h = self.fc9(h)

        self.loss = F.softmax_cross_entropy(h, t)
        self.accuracy = F.accuracy(h, t)
        return self.loss

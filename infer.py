import torch
from model import AudioToPose, PoseDiscriminator
from transforms import BVHtoMocapData, MocapDataToExpMap, Pipeline, AudioToLogMelSpec
from pathlib import Path
import librosa

class GesturePrediction:

    def __init__(self, checkpoint):
        checkpoint = Path(checkpoint)
        self.mocap_pipeline = Pipeline([BVHtoMocapData, MocapDataToExpMap])
        self.mel_spec = AudioToLogMelSpec()
        self.generator = None
        self.discriminator = None
        self.last_audio_shape = None
        self.checkpoint = torch.load(checkpoint)
        self.infer = checkpoint.parent / 'infer/'
        self.infer.mkdir(exist_ok=True)


    def apply(self, audio_file, bvh_file):
        real_pose = torch.from_numpy(self.mocap_pipeline.apply(bvh_file)).unsqueeze(0).cuda()
        audio_encoding = torch.from_numpy(self.mel_spec.apply(librosa.load(audio_file, mono=True))).unsqueeze(0).cuda()
        if self.generator is None or audio_encoding.shape[-2:] != self.last_audio_shape:
            self.generator = AudioToPose(input_shape=audio_encoding.shape[-2:], pose_shape=real_pose.shape[-2:],
                                         encoder_dim=args.encoder_dim)
            self.generator.load_state_dict(self.checkpoint['model_state_dict']['generator'])
            self.generator.float()
            self.generator.eval()
            self.last_audio_shape = audio_encoding.shape[-2:]
            # self.discriminator = PoseDiscriminator(pose_shape=real_pose.shape[-2:]).cuda()
            # self.discriminator.load_state_dict(self.checkpoint['model_state_dict']['discriminator'])
            # self.discriminator.float()
            # self.discriminator.eval()

        pred_pose = self.generator(audio_encoding)
        print(losses.l1(pred_pose, real_pose))
        bvh = self.mocap_pipeline.invert(pred_pose[0].permute(1, 0).detach().cpu().numpy())

        with open(infer / (audio_file.name.split('.')[0] + '.bvh'), 'w+') as f:
            f.write(bvh)

    @staticmethod
    def get_losses():
        """
        Returns a namespace of required loss functions.
        """
        losses = Namespace()
        losses.l1 = nn.L1Loss()
        losses.mse = nn.MSELoss()
        return losses

gen = GesturePrediction('E:/Users/Sarthak/Experiments/speech2gesture/20210929-211434-1d-unet-666343a/checkpoints/best.pt')
audio = Path('E:/Users/Sarthak/Data/speech2gesture/raw_data/val/Audio/Recording_018.wav')
bvh = Path('E:/Users/Sarthak/Data/speech2gesture/raw_data/val/Motion/Recording_018.bvh')

gen.apply(audio, bvh)
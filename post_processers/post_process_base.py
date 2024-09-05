from abc import ABC,abstractmethod


class PostProcessor(ABC):
    @abstractmethod
    def postprocess(self,text_input,count_track,background_dir,fps,frames):
        pass
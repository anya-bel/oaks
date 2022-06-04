from tempfile import TemporaryDirectory
import os
import sys
import json

class Text2Triple:

    res_name = "OIE_2016.bert-base-cased.np.d2048.b6.sorted/P0_result.json"
    command_call = "python scripts/manager.py --task=OIE_2016 --cuda=0 --model=\"bert-base-cased\" --data-dir=\"{}\" --output-dir=\"{}\" --runs-dir=\"{}\" --result-dir=\"{}\""

    def convert_sentences(self,sentences):
        with TemporaryDirectory() as tmpdir:
            tmpfile = os.sep.join((tmpdir,"data/test.txt"))

            with open(tmpfile,"w+") as f:
                for s in sentences: f.write(s + "\n")

            res = self._run_system(tmpdir)

        return res

    def _run_system(self, in_dir):
        out_dir = TemporaryDirectory()
        run_dir = TemporaryDirectory()
        res_dir = TemporaryDirectory()

        self._make_calls(in_dir,out_dir,run_dir,res_dir)

        res = self._retrieve_result(res_dir)

        out_dir.cleanup()
        run_dir.cleanup()
        res_dir.cleanup()

        return res

    def _make_calls(self, in_dir, out_dir, run_dir, res_dir):

        # Hide errors
        null = open(os.devnull,'wb')

        _stdout, sys.stdout = sys.stdout, null
        _stderr, sys.stderr = sys.stderr, null

        command_s = self.command_call.format(in_dir + "/",
                                                out_dir.name+ "/",
                                                run_dir.name+ "/",
                                                res_dir.name+ "/")
        os.system(command_s + " &> /dev/null")

        sys.stdout, sys.stderr = _stdout, _stderr

    def _retrieve_result(self,res_dir):
        json_path = os.sep.join((res_dir.name,self.res_name))

        with open(json_path,"r") as jf:
            res_dict = json.load(jf)

        return res_dict


sentences = ["Simone de Beauvoir was a French writer and feminist, a member of the intellectual fellowship of philosopher-writers who have given a literary transcription to the themes of existentialism."]
if __name__ == "__main__":
    t2t = Text2Triple()
    r = t2t.convert_sentences(sentences)
    print(r)

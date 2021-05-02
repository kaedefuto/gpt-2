from transformers import AutoModelForCausalLM
from transformers import T5Tokenizer
import csv

#入力文字
enter = ['張り地の素材は本革なので', '張り地の素材は合成皮革なので', '張り地の素材は綿なので', '張り地の素材はファブリックなので', '張り地には本革を用いているので', '張り地には合成皮革を用いているので', '張り地には綿を用いているので', '張り地にはファブリックを用いているので']
name = 'hariji'
num = 1 

header = ["入力","出力"]
with open("{0}_gen.csv".format(name), 'w', encoding="sjis") as f:
  writer = csv.writer(f, lineterminator="\n")
  writer.writerow(header)

# トークナイザーとモデルの準備
tokenizer = T5Tokenizer.from_pretrained("rinna/japanese-gpt2-medium")
model = AutoModelForCausalLM.from_pretrained("output3/")

for sentence in enter:
  print(num)
  # 推論
  input = tokenizer.encode(sentence, return_tensors="pt")
  output = model.generate(input, do_sample=True, max_length=100, num_return_sequences=25)
  outer = tokenizer.batch_decode(output)

  #</s>の削除
  outer = [s.replace('</s> ','') for s in outer]
  length = len(outer)
  # for i in range(length):
  #   output[i] = output[i].replace('</s> ','')
  #   print(output[i])

  #CSV書き込み
  header = ["入力","出力"]
  with open("{0}_gen.csv".format(name), 'a', encoding="sjis") as f:
    writer = csv.writer(f, lineterminator="\n")
    one_row = [sentence, '']
    writer.writerow(one_row)
    for i in range(length):
      one_row = ["", outer[i]]
      writer.writerow(one_row)
  num += 1

print('complete')
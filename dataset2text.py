# 이 파일은 xls 형식의 데이터셋을 bert 학습을 위한 txt 형식의 파일로 변환해주는 코드입니다.
# 형식 : id     content     sentiment
# sentiment type : 공포 놀람 분노 슬픔 중립 행복 혐오
import pandas as pd
import os

def convert(input_file, output_file):
    try:
        df = pd.read_excel(input_file)
        if 'Sentence' not in df.columns or 'Emotion' not in df.columns:
            raise ValueError("Sentence or Emotion is not in excel columns.")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("id\tsentence\tsentiment\n")

            for idx, row in df.iterrows():
                sentence = str(row['Sentence']).replace("\t", ' ').replace('\n', ' ').strip()
                sentiment = str(row['Emotion']).strip()
                f.write(f"{idx+1}\t{sentence}\t{sentiment}\n")
        
        print(f'Convert completed!')
        print(f'Total data count : {len(df)}')

    except Exception as e:
        print("error: ", e)
        raise e

def main():
    current_dir = os.getcwd()
    input_file = os.path.join(current_dir, "datasets/talk_data.xlsx")
    output_file = os.path.join(current_dir, "datasets/talk_data.txt")

    convert(input_file, output_file)

main()
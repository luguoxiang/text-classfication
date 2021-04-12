import boto3
import os
import sys
from concurrent.futures import ThreadPoolExecutor
import codecs

TEXT_ENCODING="utf-8"

session = boto3.Session(
            aws_access_key_id = os.environ["AWS_ACCESS_KEY_ID"].strip(),
            aws_secret_access_key = os.environ["AWS_SECRET_ACCESS_KEY"].strip(),
            region_name = os.environ["AWS_REGION_NAME"].strip())
s3 = session.resource('s3')

root_dir=sys.argv[1]
print("uploading data from {}...".format(root_dir))

cat_list = os.listdir(root_dir)

BLOCKSIZE = 1025 * 512
def taskf(src, dst):
    if TEXT_ENCODING == "utf-8":
        s3.Object("text-cls-data", dst).upload_file(src)
        return dst
    with codecs.open(src, "r", TEXT_ENCODING, errors='ignore') as sourceFile:
        with codecs.open(src+".utf8", "w", "utf-8") as targetFile:
            while True:
                contents = sourceFile.read(BLOCKSIZE)
                if not contents:
                    break
                targetFile.write(contents)
    s3.Object("text-cls-data", dst).upload_file(src+".utf8")
    return dst

with ThreadPoolExecutor(max_workers=20) as executor:
    for cat_index in range(len(cat_list)): # iterate category
        cat_dir = cat_list[cat_index]
        cat_path=os.path.join(root_dir, cat_dir)
        if not os.path.isdir(cat_path): continue

        process_time = 0.0
        futures = []
        for sample_file in os.listdir(cat_path): # iterate document under category
                if sample_file.endswith(".utf8"):
                    continue
                sample_path = os.path.join(cat_path, sample_file)
                if os.path.isdir(sample_path): continue

                s3_path = "news/%s/%s" % (cat_dir, sample_file)
                future = executor.submit(taskf, sample_path, s3_path)
                futures.append(future)

        print("Uploading %s(%d/%d, %d files) ...            " % (cat_dir, cat_index + 1, len(cat_list), len(futures)))
        for index in range(len(futures)):
            print("uploaded(%d/%d) %s" % (index, len(futures), futures[index].result()), end="\r")
    print("Done                       ")

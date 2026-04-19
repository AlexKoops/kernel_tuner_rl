import argparse
import SearchspaceInfo

parser = argparse.ArgumentParser()
parser.add_argument("file", help="Path to the .json or .json.gz file")
args = parser.parse_args()

searchspaceInfo = SearchspaceInfo(args.file)

counter = 3
for key in SearchspaceInfo.neighbours:
    if counter == 0:
        break
    print(searchspaceInfo.neighbours[key])

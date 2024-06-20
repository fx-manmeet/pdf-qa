import argparse
from query_pipeline import query_pipeline

def main():
    parser = argparse.ArgumentParser(description="Query the cricket event classification project report")
    parser.add_argument("pdf_file", type=str, help="The path to the PDF file to be queried")
    parser.add_argument("query", type=str, help="The query to ask about the project report")
    args = parser.parse_args()

    result = query_pipeline(args.pdf_file, args.query)
    print(result)

if __name__ == "__main__":
    main()


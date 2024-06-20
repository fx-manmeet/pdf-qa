import argparse
from query_pipeline import query_pipeline

def main():
    parser = argparse.ArgumentParser(description="Query")
    parser.add_argument("query", type=str, help="The query to ask about the given context")
    args = parser.parse_args()

    result = query_pipeline(args.query)
    print(result)

if __name__ == "__main__":
    main()

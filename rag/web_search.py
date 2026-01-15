from duckduckgo_search import DDGS

def web_search(query: str, max_results: int = 5) -> str:
    results_text = []

    with DDGS() as ddgs:
        results = ddgs.text(query, max_results=max_results)

        for r in results:
            results_text.append(r["body"])

    return "\n".join(results_text)

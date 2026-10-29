from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

# 1. Initialize the engines
analyzer = AnalyzerEngine()
anonymizer = AnonymizerEngine()

# 2. Define some text containing sample PII
text_to_test = "Hello, my name is John Doe. My phone number is 212-555-5555 and my email is john.doe@example.com."

print("--- Original Text ---")
print(text_to_test)

# 3. Analyze the text to find PII
analyzer_results = analyzer.analyze(text=text_to_test, language="en")

print("\n--- Detected PII Entities ---")
for result in analyzer_results:
    print(f"Found {result.entity_type} from character {result.start} to {result.end} (Confidence: {result.score:.2f})")

# 4. Anonymize the detected PII
anonymized_results = anonymizer.anonymize(text=text_to_test, analyzer_results=analyzer_results)

print("\n--- Anonymized Text ---")
print(anonymized_results.text)
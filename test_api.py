#!/usr/bin/env python3
"""
Test script pour l'API FastAPI avec scraping intégré
"""

import requests
import json
import time

BASE_URL = "http://localhost:8000"

def test_health():
    """Test de l'endpoint /health"""
    print("🔍 Testing /health endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            print("✅ Health check passed!")
            print(f"   Response: {response.json()}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Connection error: {e}")
        return False

def test_predict():
    """Test de l'endpoint /predict"""
    print("\n🔍 Testing /predict endpoint...")
    payload = {
        "text": "This product is amazing! I love it so much, great quality and fast delivery!"
    }
    try:
        response = requests.post(f"{BASE_URL}/predict", json=payload)
        if response.status_code == 200:
            result = response.json()
            print("✅ Prediction successful!")
            print(f"   Sentiment: {result['sentiment']}")
            print(f"   Confidence: {result['confidence']}")
            return True
        else:
            print(f"❌ Prediction failed: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"❌ Connection error: {e}")
        return False

def test_scrape():
    """Test de l'endpoint /scrape"""
    print("\n🔍 Testing /scrape endpoint...")
    payload = {
        "source": "trustpilot",
        "target": "amazon.in",
        "max_pages": 1
    }
    try:
        response = requests.post(f"{BASE_URL}/scrape", json=payload)
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print("✅ Scraping successful!")
            print(f"   Reviews found: {result['statistics']['total']}")
            print(f"   Source: {result['statistics']['source']}")
            print(f"   Positive: {result['statistics']['POSITIF']}")
            print(f"   Negative: {result['statistics']['NEGATIF']}")
            return True
        else:
            print(f"❌ Scraping failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Connection error: {e}")
        return False

def main():
    print("🚀 Testing Flipkart Sentiment Analysis API")
    print("=" * 50)

    # Attendre que le serveur soit prêt
    print("⏳ Waiting for server to be ready...")
    time.sleep(2)

    # Tests séquentiels
    tests = [
        ("Health Check", test_health),
        ("Single Prediction", test_predict),
        ("Scraping + Analysis", test_scrape)
    ]

    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        success = test_func()
        results.append((test_name, success))

    # Résumé
    print("\n" + "="*50)
    print("📊 TEST SUMMARY")
    print("="*50)
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{test_name:20} : {status}")

    total_passed = sum(1 for _, success in results if success)
    print(f"\nTotal: {total_passed}/{len(results)} tests passed")

    if total_passed == len(results):
        print("🎉 All tests passed! API is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the API logs for details.")

if __name__ == "__main__":
    main()
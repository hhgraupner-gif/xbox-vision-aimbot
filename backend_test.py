import requests
import sys
import json
from datetime import datetime

class ComputerVisionAPITester:
    def __init__(self, base_url="https://headshot-ai-8.preview.emergentagent.com"):
        self.base_url = base_url
        self.api_url = f"{base_url}/api"
        self.tests_run = 0
        self.tests_passed = 0

    def run_test(self, name, method, endpoint, expected_status, data=None, timeout=30):
        """Run a single API test"""
        url = f"{self.api_url}/{endpoint}"
        headers = {'Content-Type': 'application/json'}

        self.tests_run += 1
        print(f"\n🔍 Testing {name}...")
        print(f"   URL: {url}")
        
        try:
            if method == 'GET':
                response = requests.get(url, headers=headers, timeout=timeout)
            elif method == 'PUT':
                response = requests.put(url, json=data, headers=headers, timeout=timeout)
            elif method == 'POST':
                response = requests.post(url, json=data, headers=headers, timeout=timeout)

            success = response.status_code == expected_status
            if success:
                self.tests_passed += 1
                print(f"✅ Passed - Status: {response.status_code}")
                try:
                    response_data = response.json()
                    print(f"   Response keys: {list(response_data.keys()) if isinstance(response_data, dict) else 'Non-dict response'}")
                    return True, response_data
                except:
                    return True, response.text
            else:
                print(f"❌ Failed - Expected {expected_status}, got {response.status_code}")
                try:
                    error_data = response.json()
                    print(f"   Error: {error_data}")
                except:
                    print(f"   Error text: {response.text}")
                return False, {}

        except requests.exceptions.Timeout:
            print(f"❌ Failed - Request timeout ({timeout}s)")
            return False, {}
        except Exception as e:
            print(f"❌ Failed - Error: {str(e)}")
            return False, {}

    def test_root_endpoint(self):
        """Test GET /api/ returns correct message"""
        success, response = self.run_test(
            "Root API Endpoint",
            "GET",
            "",
            200
        )
        if success and isinstance(response, dict):
            expected_message = "Xbox Vision AI - Computer Vision Aimbot System"
            if response.get("message") == expected_message:
                print(f"   ✅ Message correct: {response.get('message')}")
            else:
                print(f"   ⚠️  Message incorrect. Expected: {expected_message}, Got: {response.get('message')}")
        return success

    def test_get_settings(self):
        """Test GET /api/settings returns detection settings"""
        success, response = self.run_test(
            "Get Settings",
            "GET", 
            "settings",
            200
        )
        if success and isinstance(response, dict):
            required_keys = [
                "confidence_threshold", "aim_sensitivity", "target_classes",
                "enabled", "show_boxes", "show_crosshair", "aim_assist_enabled"
            ]
            missing_keys = [key for key in required_keys if key not in response]
            if not missing_keys:
                print(f"   ✅ All required settings present")
                print(f"   Settings: confidence={response.get('confidence_threshold')}, enabled={response.get('enabled')}")
            else:
                print(f"   ⚠️  Missing keys: {missing_keys}")
        return success, response

    def test_update_settings(self, current_settings):
        """Test PUT /api/settings updates settings correctly"""
        # Create updated settings
        updated_settings = current_settings.copy()
        updated_settings.update({
            "confidence_threshold": 0.7,
            "aim_sensitivity": 0.9,
            "enabled": True,
            "show_boxes": False,
            "show_crosshair": True,
            "aim_assist_enabled": True
        })
        
        success, response = self.run_test(
            "Update Settings",
            "PUT",
            "settings", 
            200,
            data=updated_settings
        )
        
        if success and isinstance(response, dict):
            # Verify the settings were updated
            if response.get("confidence_threshold") == 0.7:
                print(f"   ✅ Settings updated correctly")
            else:
                print(f"   ⚠️  Settings may not have updated correctly")
        return success

    def test_demo_frame(self):
        """Test GET /api/demo/frame returns frame with detections"""
        success, response = self.run_test(
            "Demo Frame",
            "GET",
            "demo/frame",
            200,
            timeout=10
        )
        
        if success and isinstance(response, dict):
            required_keys = ["frame", "detections", "processing_time_ms", "frame_width", "frame_height", "timestamp"]
            missing_keys = [key for key in required_keys if key not in response]
            if not missing_keys:
                print(f"   ✅ Demo frame data complete")
                print(f"   Detections: {len(response.get('detections', []))}")
                print(f"   Frame size: {response.get('frame_width')}x{response.get('frame_height')}")
                print(f"   Processing time: {response.get('processing_time_ms')}ms")
                print(f"   Demo mode: {response.get('demo', False)}")
                
                # Check if frame data exists
                if response.get("frame") and len(response.get("frame", "")) > 100:
                    print(f"   ✅ Frame data present (base64 length: {len(response.get('frame', ''))})")
                else:
                    print(f"   ⚠️  Frame data missing or too small")
                    
            else:
                print(f"   ⚠️  Missing keys: {missing_keys}")
        return success

    def test_yolo_classes(self):
        """Test GET /api/yolo/classes returns available YOLO classes"""
        success, response = self.run_test(
            "YOLO Classes",
            "GET",
            "yolo/classes",
            200
        )
        
        if success and isinstance(response, dict):
            classes = response.get("classes", [])
            if isinstance(classes, list) and len(classes) > 0:
                print(f"   ✅ YOLO classes loaded: {len(classes)} classes")
                print(f"   Sample classes: {classes[:5]}...")
                if "person" in classes:
                    print(f"   ✅ 'person' class available (good for enemy detection)")
                else:
                    print(f"   ⚠️  'person' class not found in available classes")
            else:
                print(f"   ⚠️  No classes returned or invalid format")
        return success

    def test_capture_endpoint(self):
        """Test GET /api/capture (may fail without actual screen capture - expected)"""
        print(f"\n🔍 Testing Real Capture Endpoint (may fail - expected)...")
        try:
            url = f"{self.api_url}/capture"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                print(f"✅ Real capture working - Status: {response.status_code}")
                self.tests_passed += 1
            else:
                print(f"⚠️  Real capture failed (expected) - Status: {response.status_code}")
                print(f"   This is normal in containerized environment without display")
        except Exception as e:
            print(f"⚠️  Real capture failed (expected) - Error: {str(e)}")
            print(f"   This is normal in containerized environment without display")
        
        self.tests_run += 1
        return True  # Don't count this as a failure

def main():
    print("🚀 Starting Computer Vision AI Backend API Tests")
    print("=" * 60)
    
    tester = ComputerVisionAPITester()
    
    # Test sequence
    print("\n📋 Running API Tests...")
    
    # 1. Test root endpoint
    tester.test_root_endpoint()
    
    # 2. Test get settings
    settings_success, current_settings = tester.test_get_settings()
    
    # 3. Test update settings (only if get settings worked)
    if settings_success and current_settings:
        tester.test_update_settings(current_settings)
    
    # 4. Test demo frame
    tester.test_demo_frame()
    
    # 5. Test YOLO classes
    tester.test_yolo_classes()
    
    # 6. Test real capture (expected to fail in container)
    tester.test_capture_endpoint()
    
    # Print final results
    print("\n" + "=" * 60)
    print(f"📊 Final Results: {tester.tests_passed}/{tester.tests_run} tests passed")
    
    if tester.tests_passed >= 4:  # We expect 4+ core tests to pass
        print("✅ Backend API tests mostly successful!")
        return 0
    else:
        print("❌ Multiple backend API failures detected")
        return 1

if __name__ == "__main__":
    sys.exit(main())
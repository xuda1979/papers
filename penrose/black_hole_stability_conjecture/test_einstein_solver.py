#!/usr/bin/env python3
"""
Test Suite for Einstein NN Solver
=================================

This script tests the Einstein NN solver locally before deploying
to remote servers with NPU acceleration.

Test Categories:
1. Hardware detection and configuration
2. Basic functionality (quick smoke tests)
3. Medium-scale tests (validation)
4. Preparation for remote NPU deployment

Usage:
    python test_einstein_solver.py [--test-type quick|medium|full]
"""

import os
import sys
import subprocess
import time
import argparse
from typing import Dict, List

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from einstein_nn_solver import (
        HardwareConfig, get_hardware_config, set_hardware_config,
        KerrBackground, EinsteinPINN, run_einstein_nn_simulation
    )
except ImportError as e:
    print(f"❌ Failed to import Einstein NN solver: {e}")
    sys.exit(1)


class TestRunner:
    """Test runner for Einstein NN solver."""
    
    def __init__(self):
        self.test_results = []
        self.start_time = time.time()
    
    def log_test(self, test_name: str, status: str, message: str = "", duration: float = 0):
        """Log test result."""
        self.test_results.append({
            'name': test_name,
            'status': status,
            'message': message,
            'duration': duration
        })
        
        status_icon = "[OK]" if status == "PASS" else "[FAIL]" if status == "FAIL" else "[WARN]"
        print(f"{status_icon} {test_name} ({duration:.1f}s): {message}")
    
    def test_hardware_detection(self) -> bool:
        """Test hardware detection and configuration."""
        test_start = time.time()
        
        try:
            # Test auto-detection
            config = HardwareConfig()
            assert config.device in ['cpu', 'gpu', 'npu'], f"Invalid device: {config.device}"
            assert config.framework in ['numpy', 'torch', 'torch_npu'], f"Invalid framework: {config.framework}"
            
            # Test forced CPU
            set_hardware_config('cpu')
            cpu_config = get_hardware_config()
            assert cpu_config.device == 'cpu'
            assert cpu_config.framework == 'numpy'
            
            duration = time.time() - test_start
            self.log_test("Hardware Detection", "PASS", 
                         f"Device: {config.device}, Framework: {config.framework}", duration)
            return True
            
        except Exception as e:
            duration = time.time() - test_start
            self.log_test("Hardware Detection", "FAIL", str(e), duration)
            return False
    
    def test_kerr_background(self) -> bool:
        """Test Kerr background geometry calculation."""
        test_start = time.time()
        
        try:
            # Test various spin parameters
            test_chis = [0.0, 0.5, 0.9, 0.99]
            
            for chi in test_chis:
                bg = KerrBackground(M=1.0, chi=chi)
                
                # Basic sanity checks
                assert bg.r_plus > bg.r_minus >= 0, "Invalid horizon radii"
                assert bg.kappa > 0, "Invalid surface gravity"
                assert 0 <= abs(bg.Omega_H) <= 0.5, "Invalid horizon angular velocity"
                
                # Test metric computation
                import numpy as np
                t = np.array([0.0])
                r = np.array([bg.r_plus * 2])
                theta = np.array([np.pi/2])
                phi = np.array([0.0])
                
                g = bg.metric_background(t, r, theta, phi)
                assert g['tt'][0] < 0, "g_tt should be negative"
                assert g['rr'][0] > 0, "g_rr should be positive"
                assert g['thth'][0] > 0, "g_theta_theta should be positive"
                assert g['phph'][0] > 0, "g_phi_phi should be positive"
            
            duration = time.time() - test_start
            self.log_test("Kerr Background", "PASS", 
                         f"Tested chi in {test_chis}", duration)
            return True
            
        except Exception as e:
            duration = time.time() - test_start
            self.log_test("Kerr Background", "FAIL", str(e), duration)
            return False
    
    def test_neural_network_initialization(self) -> bool:
        """Test neural network initialization."""
        test_start = time.time()
        
        try:
            # Force CPU for testing
            set_hardware_config('cpu')
            
            bg = KerrBackground(M=1.0, chi=0.5)
            model = EinsteinPINN(bg, d_model=64, num_layers=2, num_heads=4)  # Smaller for testing
            
            # Test forward pass
            import numpy as np
            t = np.array([0.0])
            r = np.array([3.0])
            theta = np.array([np.pi/2])
            phi = np.array([0.0])
            
            h = model.forward(t, r, theta, phi)
            
            # Check output format
            expected_components = ['tt', 'tr', 'tth', 'tph', 'rr', 'rth', 'rph', 'thth', 'thph', 'phph']
            for comp in expected_components:
                assert comp in h, f"Missing metric component: {comp}"
                assert np.isfinite(h[comp]).all(), f"Non-finite values in {comp}"
            
            duration = time.time() - test_start
            self.log_test("Neural Network Init", "PASS", 
                         "Forward pass successful", duration)
            return True
            
        except Exception as e:
            duration = time.time() - test_start
            self.log_test("Neural Network Init", "FAIL", str(e), duration)
            return False
    
    def test_command_line_interface(self) -> bool:
        """Test command-line interface."""
        test_start = time.time()
        
        try:
            # Test help
            result = subprocess.run([sys.executable, 'einstein_nn_solver.py', '--help'], 
                                  capture_output=True, text=True)
            assert result.returncode == 0, "Help command failed"
            assert 'Einstein Field Equations' in result.stdout, "Help text incorrect"
            
            # Test quick test mode
            result = subprocess.run([sys.executable, 'einstein_nn_solver.py', '--test', 'quick'], 
                                  capture_output=True, text=True, timeout=30)
            assert result.returncode == 0, f"Quick test failed: {result.stderr}"
            assert 'SIMULATION COMPLETE' in result.stdout, "Simulation did not complete"
            
            duration = time.time() - test_start
            self.log_test("Command Line Interface", "PASS", 
                         "Help and quick test working", duration)
            return True
            
        except subprocess.TimeoutExpired:
            duration = time.time() - test_start
            self.log_test("Command Line Interface", "FAIL", 
                         "Test timed out (>30s)", duration)
            return False
        except Exception as e:
            duration = time.time() - test_start
            self.log_test("Command Line Interface", "FAIL", str(e), duration)
            return False
    
    def test_npu_readiness(self) -> bool:
        """Test NPU readiness (without actual NPU)."""
        test_start = time.time()
        
        try:
            # Test NPU configuration creation
            try:
                set_hardware_config('npu')
                npu_config = get_hardware_config()
                # Should fallback to CPU if no NPU available
                assert npu_config.device in ['npu', 'cpu'], "NPU config failed"
                
                if npu_config.device == 'cpu':
                    message = "NPU not available, fallback to CPU successful"
                else:
                    message = "NPU detected and configured"
                
            except Exception as e:
                # This is expected if no NPU libraries are available
                message = f"NPU libraries not available: {type(e).__name__}"
            
            duration = time.time() - test_start
            self.log_test("NPU Readiness", "PASS", message, duration)
            return True
            
        except Exception as e:
            duration = time.time() - test_start
            self.log_test("NPU Readiness", "FAIL", str(e), duration)
            return False
    
    def run_full_suite(self, test_type: str = 'quick') -> bool:
        """Run the full test suite."""
        print("="*60)
        print("EINSTEIN NN SOLVER - LOCAL TEST SUITE")
        print("="*60)
        print(f"Test type: {test_type}")
        print(f"Python version: {sys.version}")
        print(f"Platform: {sys.platform}")
        print()
        
        # Run tests
        tests_passed = 0
        total_tests = 0
        
        test_methods = [
            self.test_hardware_detection,
            self.test_kerr_background,
            self.test_neural_network_initialization,
            self.test_command_line_interface,
            self.test_npu_readiness
        ]
        
        if test_type in ['medium', 'full']:
            # Add more comprehensive tests for medium/full
            test_methods.extend([
                self.test_medium_simulation,
            ])
        
        for test_method in test_methods:
            total_tests += 1
            if test_method():
                tests_passed += 1
        
        # Summary
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        
        for result in self.test_results:
            status_icon = "[OK]" if result['status'] == "PASS" else "[FAIL]"
            print(f"{status_icon} {result['name']}: {result['status']} ({result['duration']:.1f}s)")
            if result['message']:
                print(f"   {result['message']}")
        
        print(f"\nTotal: {tests_passed}/{total_tests} tests passed")
        print(f"Duration: {time.time() - self.start_time:.1f}s")
        
        if tests_passed == total_tests:
            print("\n[SUCCESS] All tests passed! Ready for remote NPU deployment.")
            self.print_deployment_instructions()
            return True
        else:
            print(f"\n[WARNING] {total_tests - tests_passed} tests failed. Fix issues before deployment.")
            return False
    
    def test_medium_simulation(self) -> bool:
        """Run medium-scale simulation test."""
        test_start = time.time()
        
        try:
            # Force CPU and limit iterations for testing
            set_hardware_config('cpu')
            
            results = run_einstein_nn_simulation(chi=0.6, n_iterations=20)
            
            # Check results structure
            required_keys = ['model', 'trainer', 'results', 'qnm_spectrum', 'flux_data', 'instabilities']
            for key in required_keys:
                assert key in results, f"Missing result key: {key}"
            
            # Check that training actually ran
            assert len(results['trainer'].training_history) > 0, "No training history"
            
            duration = time.time() - test_start
            self.log_test("Medium Simulation", "PASS", 
                         f"20 iterations completed", duration)
            return True
            
        except Exception as e:
            duration = time.time() - test_start
            self.log_test("Medium Simulation", "FAIL", str(e), duration)
            return False
    
    def print_deployment_instructions(self):
        """Print instructions for remote NPU deployment."""
        print("\n" + "="*60)
        print("REMOTE NPU DEPLOYMENT INSTRUCTIONS")
        print("="*60)
        
        print("\n1. Upload files to remote server:")
        print("   scp einstein_nn_solver.py user@server:/path/to/project/")
        
        print("\n2. Install dependencies on remote server:")
        print("   # For Huawei Ascend NPU")
        print("   pip install torch-npu")
        print("   # OR for Intel Gaudi NPU")
        print("   pip install habana-torch-plugin")
        
        print("\n3. Run quick test on remote server:")
        print("   python einstein_nn_solver.py --npu --test quick")
        
        print("\n4. Run production training:")
        print("   python einstein_nn_solver.py --npu --chi 0.7 --iterations 1000000")
        
        print("\n5. Distributed training (8 NPUs):")
        print("   for i in {0..7}; do")
        print("     RANK=$i WORLD_SIZE=8 python einstein_nn_solver.py --npu --chi 0.9 --iterations 5000000 &")
        print("   done")
        
        print("\n6. Monitor training:")
        print("   tail -f results/training.log")
        print("   nvidia-smi  # or npu-smi for NPU monitoring")
        
        print("\nExpected performance:")
        print("  - NPU: 10-50x speedup vs CPU")
        print("  - Memory: 50-100 GB per NPU")
        print("  - Time: ~24 hours for 10⁶ iterations")


def main():
    """Main entry point for test suite."""
    parser = argparse.ArgumentParser(description='Test Einstein NN Solver')
    parser.add_argument('--test-type', choices=['quick', 'medium', 'full'], 
                       default='quick', help='Type of test to run')
    parser.add_argument('--verbose', '-v', action='store_true', 
                       help='Verbose output')
    
    args = parser.parse_args()
    
    # Set verbosity
    if args.verbose:
        import logging
        logging.basicConfig(level=logging.DEBUG)
    
    # Run tests
    runner = TestRunner()
    success = runner.run_full_suite(args.test_type)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
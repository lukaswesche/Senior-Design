import SwiftUI
import UIKit
import AVFoundation
import Vision
import CoreImage
import Combine

// MARK: - ContentView: Displays the camera preview and performs analysis.
struct ContentView: UIViewControllerRepresentable {
    @Binding var pipetDiameter: String
    @Binding var density: String   // Density in kg/m³

    // MARK: - Coordinator
    class Coordinator: NSObject, AVCapturePhotoCaptureDelegate, ObservableObject {
        let session = AVCaptureSession()
        let output = AVCapturePhotoOutput()
        var parent: ContentView
        @Published var surfaceTension: String = "Tap to measure"
        var photoCompletion: ((UIImage) -> Void)?
        var cancellables = Set<AnyCancellable>()
        
        // Keep a reference to the preview layer (needed for tap-to-focus).
        var previewLayer: AVCaptureVideoPreviewLayer?
        
        init(parent: ContentView) {
            self.parent = parent
            super.init()
            setupCamera()
            checkPermissions()
        }
        
        func setupCamera() {
            session.sessionPreset = .high
            guard let device = AVCaptureDevice.default(.builtInWideAngleCamera,
                                                         for: .video,
                                                         position: .back),
                  let input = try? AVCaptureDeviceInput(device: device) else {
                print("Failed to get camera input")
                return
            }
            
            // Configure focus and exposure for extremely close-up capture.
            do {
                try device.lockForConfiguration()
                // Center focus and exposure.
                if device.isFocusModeSupported(.continuousAutoFocus) {
                    device.focusPointOfInterest = CGPoint(x: 0.5, y: 0.5)
                    device.focusMode = .continuousAutoFocus
                }
                if device.isExposureModeSupported(.continuousAutoExposure) {
                    device.exposurePointOfInterest = CGPoint(x: 0.5, y: 0.5)
                    device.exposureMode = .continuousAutoExposure
                }
                // For super close-up, set a very low lens position.
                if device.isLockingFocusWithCustomLensPositionSupported {
                    // API expects a Float; we directly declare desiredLensPosition as a Float.
                    let desiredLensPosition: Float = 0.05
                    device.setFocusModeLocked(lensPosition: desiredLensPosition, completionHandler: nil)
                }
                device.unlockForConfiguration()
            } catch {
                print("Error configuring device: \(error)")
            }
            
            if session.canAddInput(input) {
                session.addInput(input)
            }
            if session.canAddOutput(output) {
                session.addOutput(output)
            }
        }
        
        func checkPermissions() {
            let status = AVCaptureDevice.authorizationStatus(for: .video)
            switch status {
            case .authorized:
                DispatchQueue.global(qos: .userInitiated).async {
                    self.session.startRunning()
                }
            case .notDetermined:
                AVCaptureDevice.requestAccess(for: .video) { granted in
                    if granted {
                        DispatchQueue.global(qos: .userInitiated).async {
                            self.session.startRunning()
                        }
                    } else {
                        DispatchQueue.main.async {
                            self.surfaceTension = "Camera access denied"
                        }
                    }
                }
            default:
                self.surfaceTension = "Camera access denied"
            }
        }
        
        @objc func capturePhoto() {
            print("capturePhoto triggered")
            let settings = AVCapturePhotoSettings()
            photoCompletion = { (image: UIImage) in
                DispatchQueue.main.async {
                    let result = self.calculateSurfaceTension(from: image,
                                                              pipetDiameter: self.parent.pipetDiameter,
                                                              density: self.parent.density)
                    print("Analysis result: \(String(describing: result))")
                    self.surfaceTension = result ?? "Measurement failed"
                }
            }
            output.capturePhoto(with: settings, delegate: self)
        }
        
        func photoOutput(_ output: AVCapturePhotoOutput,
                         didFinishProcessingPhoto photo: AVCapturePhoto,
                         error: Error?) {
            if let error = error {
                print("Error capturing photo: \(error.localizedDescription)")
                return
            }
            guard let data = photo.fileDataRepresentation(),
                  let image = UIImage(data: data) else {
                print("Failed to convert image data")
                return
            }
            photoCompletion?(image)
        }
        
        /// Uses Vision to detect the droplet, computes a scale factor from pipet diameter,
        /// and calculates surface tension using the provided density.
        func calculateSurfaceTension(from image: UIImage,
                                     pipetDiameter: String,
                                     density: String) -> String? {
            guard let cgImage = image.cgImage else {
                print("No CGImage found in image")
                return nil
            }
            
            let request = VNDetectContoursRequest()
            request.contrastAdjustment = 1.0
            request.detectDarkOnLight = true
            let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
            
            do {
                try handler.perform([request])
                guard let observation = request.results?.first as? VNContoursObservation else {
                    print("No contour observation found")
                    return nil
                }
                let contours = observation.topLevelContours
                guard let droplet = contours.max(by: { $0.pointCount < $1.pointCount }) else {
                    print("No droplet contour found")
                    return nil
                }
                let points = droplet.normalizedPoints
                guard let minY = points.map({ CGFloat($0.y) }).min(),
                      let maxY = points.map({ CGFloat($0.y) }).max(),
                      let minX = points.map({ CGFloat($0.x) }).min(),
                      let maxX = points.map({ CGFloat($0.x) }).max() else {
                    print("Failed to determine bounding box")
                    return nil
                }
                
                let normalizedWidth: CGFloat = maxX - minX
                let normalizedHeight: CGFloat = maxY - minY
                
                // Compute a scale factor (mm per pixel). Default: 0.01 mm/pixel if pipet diameter is invalid.
                var scaleFactor: CGFloat = 0.01
                if let d = Double(pipetDiameter), d > 0 {
                    let measuredWidthPixels = normalizedWidth * CGFloat(cgImage.width)
                    scaleFactor = CGFloat(d) / measuredWidthPixels
                }
                
                // Convert normalized points to pixel coordinates.
                let denormPoints = points.map { point in
                    CGPoint(x: CGFloat(point.x) * CGFloat(cgImage.width),
                            y: CGFloat(point.y) * CGFloat(cgImage.height))
                }
                // Compute droplet area (pixel²) using the Shoelace formula.
                let areaPixels = polygonArea(points: denormPoints)
                // Convert area to mm².
                let areaMM = areaPixels * scaleFactor * scaleFactor
                
                // Derive effective diameter: d_eff = 2 * sqrt(area / π)
                let effectiveDiameterMM = 2 * sqrt(areaMM / CGFloat.pi)
                let effectiveDiameterM = Double(effectiveDiameterMM) / 1000.0
                
                print("Effective droplet diameter: \(effectiveDiameterMM) mm")
                
                // Use provided density, or default to water (1000 kg/m³).
                let rho: Double = (Double(density) ?? 1000.0) > 0 ? Double(density)! : 1000.0
                let g = 9.81  // m/s²
                
                // Calculate surface tension based on the effective diameter.
                let tension = rho * g * effectiveDiameterM * effectiveDiameterM  // in N/m
                
                return String(format: "Surface Tension: %.1f mN/m", tension * 1000)
            } catch {
                print("Error performing contour request: \(error.localizedDescription)")
                return nil
            }
        }
        
        /// Computes the area of a polygon using the Shoelace formula.
        func polygonArea(points: [CGPoint]) -> CGFloat {
            guard points.count > 2 else { return 0 }
            var area: CGFloat = 0
            for i in 0..<points.count {
                let j = (i + 1) % points.count
                area += points[i].x * points[j].y - points[j].x * points[i].y
            }
            return abs(area) / 2.0
        }
        
        /// Toggles the flashlight (torch) on or off.
        @objc func toggleFlashlight() {
            guard let device = AVCaptureDevice.default(for: .video), device.hasTorch else { return }
            do {
                try device.lockForConfiguration()
                if device.torchMode == .on {
                    device.torchMode = .off
                } else {
                    try device.setTorchModeOn(level: 1.0)
                }
                device.unlockForConfiguration()
            } catch {
                print("Error toggling flashlight: \(error)")
            }
        }
    } // End Coordinator

    func makeCoordinator() -> Coordinator {
        return Coordinator(parent: self)
    }
    
    func makeUIViewController(context: UIViewControllerRepresentableContext<ContentView>) -> UIViewController {
        let vc = UIViewController()
        
        // Set up the camera preview layer.
        let previewLayer = AVCaptureVideoPreviewLayer(session: context.coordinator.session)
        previewLayer.videoGravity = .resizeAspectFill
        previewLayer.frame = vc.view.bounds
        vc.view.layer.addSublayer(previewLayer)
        context.coordinator.previewLayer = previewLayer
        
        // Add tap gesture recognizer for tap-to-focus/zoom.
        let tapRecognizer = UITapGestureRecognizer(target: context.coordinator, action: #selector(Coordinator.handleTap(_:)))
        vc.view.addGestureRecognizer(tapRecognizer)
        
        // Create a button to capture a photo.
        let analysisButton = UIButton(type: .system)
        analysisButton.setTitle("Start Analysis", for: .normal)
        analysisButton.backgroundColor = UIColor.systemBlue
        analysisButton.setTitleColor(UIColor.white, for: .normal)
        analysisButton.layer.cornerRadius = 8
        analysisButton.clipsToBounds = true
        analysisButton.addTarget(context.coordinator,
                                 action: #selector(Coordinator.capturePhoto),
                                 for: .touchUpInside)
        analysisButton.translatesAutoresizingMaskIntoConstraints = false
        vc.view.addSubview(analysisButton)
        
        NSLayoutConstraint.activate([
            analysisButton.leadingAnchor.constraint(equalTo: vc.view.leadingAnchor, constant: 20),
            analysisButton.trailingAnchor.constraint(equalTo: vc.view.trailingAnchor, constant: -20),
            analysisButton.bottomAnchor.constraint(equalTo: vc.view.bottomAnchor, constant: -120),
            analysisButton.heightAnchor.constraint(equalToConstant: 40)
        ])
        
        // Create a flashlight toggle button.
        let flashlightButton = UIButton(type: .system)
        flashlightButton.setTitle("Toggle Flashlight", for: .normal)
        flashlightButton.backgroundColor = UIColor.systemYellow
        flashlightButton.setTitleColor(UIColor.black, for: .normal)
        flashlightButton.layer.cornerRadius = 8
        flashlightButton.clipsToBounds = true
        flashlightButton.addTarget(context.coordinator,
                                     action: #selector(Coordinator.toggleFlashlight),
                                     for: .touchUpInside)
        flashlightButton.translatesAutoresizingMaskIntoConstraints = false
        vc.view.addSubview(flashlightButton)
        
        NSLayoutConstraint.activate([
            flashlightButton.leadingAnchor.constraint(equalTo: vc.view.leadingAnchor, constant: 20),
            flashlightButton.trailingAnchor.constraint(equalTo: vc.view.trailingAnchor, constant: -20),
            flashlightButton.bottomAnchor.constraint(equalTo: analysisButton.topAnchor, constant: -10),
            flashlightButton.heightAnchor.constraint(equalToConstant: 40)
        ])
        
        // Create a label to display the measurement result.
        let label = UILabel()
        label.textAlignment = .center
        label.textColor = UIColor.white
        label.backgroundColor = UIColor.black.withAlphaComponent(0.7)
        label.layer.cornerRadius = 8
        label.clipsToBounds = true
        label.translatesAutoresizingMaskIntoConstraints = false
        vc.view.addSubview(label)
        
        NSLayoutConstraint.activate([
            label.leadingAnchor.constraint(equalTo: vc.view.leadingAnchor, constant: 20),
            label.trailingAnchor.constraint(equalTo: vc.view.trailingAnchor, constant: -20),
            label.bottomAnchor.constraint(equalTo: vc.view.bottomAnchor, constant: -60),
            label.heightAnchor.constraint(equalToConstant: 40)
        ])
        
        context.coordinator.$surfaceTension
            .receive(on: DispatchQueue.main)
            .sink { text in
                label.text = text
            }
            .store(in: &context.coordinator.cancellables)
        
        return vc
    }
    
    func updateUIViewController(_ uiViewController: UIViewController,
                                context: UIViewControllerRepresentableContext<ContentView>) {
        // No dynamic updates needed.
    }
}

// MARK: - Tap Handler Extension
extension ContentView.Coordinator {
    @objc func handleTap(_ sender: UITapGestureRecognizer) {
        guard let view = sender.view else { return }
        let location = sender.location(in: view)
        focusAndZoom(at: location, in: view)
    }
    
    func focusAndZoom(at point: CGPoint, in view: UIView) {
        guard let device = AVCaptureDevice.default(for: .video),
              let previewLayer = self.previewLayer else { return }
        let focusPoint = previewLayer.captureDevicePointConverted(fromLayerPoint: point)
        
        do {
            try device.lockForConfiguration()
            if device.isFocusPointOfInterestSupported {
                device.focusPointOfInterest = focusPoint
                device.focusMode = .autoFocus
            }
            if device.isExposurePointOfInterestSupported {
                device.exposurePointOfInterest = focusPoint
                device.exposureMode = .continuousAutoExposure
            }
            // Set a zoom factor (e.g., 3x).
            let desiredZoomFactor: CGFloat = 3.0
            let maxZoom = device.activeFormat.videoMaxZoomFactor
            device.videoZoomFactor = min(desiredZoomFactor, maxZoom)
            device.unlockForConfiguration()
        } catch {
            print("Error setting focus/zoom: \(error)")
        }
    }
}

// MARK: - MainView: Provides input fields and embeds ContentView.
struct MainView: View {
    @State private var pipetDiameter: String = ""
    @State private var density: String = ""
    
    var body: some View {
        VStack {
            TextField("Enter pipet diameter (mm)", text: $pipetDiameter)
                .textFieldStyle(RoundedBorderTextFieldStyle())
                .padding()
                .keyboardType(.decimalPad)
                .toolbar {
                    ToolbarItemGroup(placement: .keyboard) {
                        Spacer()
                        Button("Done") {
                            UIApplication.shared.sendAction(#selector(UIResponder.resignFirstResponder),
                                                              to: nil, from: nil, for: nil)
                        }
                    }
                }
            TextField("Enter density (kg/m³)", text: $density)
                .textFieldStyle(RoundedBorderTextFieldStyle())
                .padding([.horizontal, .bottom])
                .keyboardType(.decimalPad)
                .toolbar {
                    ToolbarItemGroup(placement: .keyboard) {
                        Spacer()
                        Button("Done") {
                            UIApplication.shared.sendAction(#selector(UIResponder.resignFirstResponder),
                                                              to: nil, from: nil, for: nil)
                        }
                    }
                }
            ContentView(pipetDiameter: $pipetDiameter, density: $density)
                .edgesIgnoringSafeArea(.all)
        }
    }
}

struct MainView_Previews: PreviewProvider {
    static var previews: some View {
        MainView()
    }
}
  

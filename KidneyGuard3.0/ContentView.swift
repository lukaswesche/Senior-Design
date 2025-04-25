// MARK: - Framework Imports
import SwiftUI            // SwiftUI for building declarative interfaces
import UIKit               // UIKit to host UIViewController in SwiftUI
import AVFoundation        // AVFoundation for camera capture and torch control
import Vision              // Vision for contour detection
import CoreImage           // CoreImage for image data handling
import Combine             // Combine for reactive bindings

// MARK: - ContentView: Hosts a UIViewController for camera preview and analysis
struct ContentView: UIViewControllerRepresentable {
    // User-provided inputs bound from SwiftUI
    @Binding var pipetDiameter: String   // Diameter of pipette or straw (mm)
    @Binding var density: String         // Liquid density (kg/m³)

    // MARK: - Coordinator: Manages camera session, photo capture, and analysis
    class Coordinator: NSObject, AVCapturePhotoCaptureDelegate, ObservableObject {
        // Camera session and photo output
        let session = AVCaptureSession()
        let output = AVCapturePhotoOutput()
        var parent: ContentView                 // Back-reference to ContentView

        // Published property to update UI with measurement results
        @Published var surfaceTension: String = "Tap to measure"

        // Closure to handle captured UIImage
        private var photoCompletion: ((UIImage) -> Void)?
        var cancellables = Set<AnyCancellable>() // Store Combine subscriptions

        // Preview layer reference for tap-to-focus and zoom
        var previewLayer: AVCaptureVideoPreviewLayer?

        // MARK: Initializer
        init(parent: ContentView) {
            self.parent = parent
            super.init()
            setupCamera()      // Configure session input/output and device settings
            checkPermissions() // Request camera access and start session
        }

        // MARK: Camera Setup
        func setupCamera() {
            session.sessionPreset = .high  // High-resolution capture

            // Select back wide-angle camera
            guard let device = AVCaptureDevice.default(
                    .builtInWideAngleCamera,
                    for: .video,
                    position: .back),
                  let input = try? AVCaptureDeviceInput(device: device)
            else {
                print("Failed to get camera input")
                return
            }

            // Lock configuration to set focus/exposure and custom lens position
            do {
                try device.lockForConfiguration()
                // Continuous auto-focus at center
                if device.isFocusModeSupported(.continuousAutoFocus) {
                    device.focusMode = .continuousAutoFocus
                    device.focusPointOfInterest = CGPoint(x: 0.5, y: 0.5)
                }
                // Continuous auto-exposure at center
                if device.isExposureModeSupported(.continuousAutoExposure) {
                    device.exposureMode = .continuousAutoExposure
                    device.exposurePointOfInterest = CGPoint(x: 0.5, y: 0.5)
                }
                // For very close-up, lock lens position if supported
                if device.isLockingFocusWithCustomLensPositionSupported {
                    let desiredLensPosition: Float = 0.05
                    device.setFocusModeLocked(lensPosition: desiredLensPosition, completionHandler: nil)
                }
                device.unlockForConfiguration()
            } catch {
                print("Error configuring device: \(error)")
            }

            // Add input and output to the session
            if session.canAddInput(input) {
                session.addInput(input)
            }
            if session.canAddOutput(output) {
                session.addOutput(output)
            }
        }

        // MARK: Permissions and Session Control
        func checkPermissions() {
            let status = AVCaptureDevice.authorizationStatus(for: .video)
            switch status {
            case .authorized:
                // Start session if already authorized
                DispatchQueue.global(qos: .userInitiated).async {
                    self.session.startRunning()
                }
            case .notDetermined:
                // Request access if not determined
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
                // Denied or restricted
                self.surfaceTension = "Camera access denied"
            }
        }

        // MARK: Photo Capture
        @objc func capturePhoto() {
            print("capturePhoto triggered")
            let settings = AVCapturePhotoSettings()

            // Define completion to process captured image
            photoCompletion = { image in
                DispatchQueue.main.async {
                    // Perform analysis using current inputs
                    let result = self.calculateSurfaceTension(
                        from: image,
                        pipetDiameter: self.parent.pipetDiameter,
                        density: self.parent.density)
                    print("Analysis result: \(String(describing: result))")
                    self.surfaceTension = result ?? "Measurement failed"
                }
            }

            // Trigger the photo capture
            output.capturePhoto(with: settings, delegate: self)
        }

        // Delegate callback with processed photo
        func photoOutput(
            _ output: AVCapturePhotoOutput,
            didFinishProcessingPhoto photo: AVCapturePhoto,
            error: Error?
        ) {
            if let error = error {
                print("Error capturing photo: \(error.localizedDescription)")
                return
            }
            // Convert data to UIImage
            guard
                let data = photo.fileDataRepresentation(),
                let image = UIImage(data: data)
            else {
                print("Failed to convert image data")
                return
            }
            // Call the completion handler
            photoCompletion?(image)
        }

        // MARK: Surface Tension Analysis
        /**
         1. Detect droplet contour via Vision.
         2. Compute mm-per-pixel scale using user pipetDiameter.
         3. Compute droplet area (shoelace formula), convert to effective diameter.
         4. Apply surface tension formula: γ = ρ·g·d_eff².
         */
        func calculateSurfaceTension(
            from image: UIImage,
            pipetDiameter: String,
            density: String
        ) -> String? {
            // Ensure CGImage available for Vision
            guard let cgImage = image.cgImage else {
                print("No CGImage found")
                return nil
            }

            // Configure contour detection request
            let request = VNDetectContoursRequest()
            request.contrastAdjustment = 1.0
            request.detectDarkOnLight = true
            let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])

            do {
                // Perform Vision request
                try handler.perform([request])
                guard let obs = request.results?.first as? VNContoursObservation else {
                    print("No contour results")
                    return nil
                }
                // Choose the largest contour (droplet)
                let contours = obs.topLevelContours
                guard let droplet = contours.max(by: { $0.pointCount < $1.pointCount }) else {
                    print("No droplet contour found")
                    return nil
                }
                let normalizedPoints = droplet.normalizedPoints

                // Convert normalized points to actual pixel coordinates
                let pixelPoints = normalizedPoints.map { point in
                    CGPoint(
                        x: CGFloat(point.x) * CGFloat(cgImage.width),
                        y: CGFloat(point.y) * CGFloat(cgImage.height)
                    )
                }

                // Compute polygon area via Shoelace formula
                let areaPixels = polygonArea(points: pixelPoints)

                // Determine scale factor (mm per pixel)
                let scale: CGFloat = {
                    guard let d = Double(pipetDiameter), d > 0 else { return 0.01 }
                    // Use bounding box width for scale
                    let widths = normalizedPoints.map { CGFloat($0.x) }
                    let minX = widths.min()!, maxX = widths.max()!
                    let pixelWidth = (maxX - minX) * CGFloat(cgImage.width)
                    return CGFloat(d) / pixelWidth
                }()

                // Convert area to mm²
                let areaMM2 = areaPixels * scale * scale
                // Compute effective diameter in mm: d_eff = 2 * sqrt(area/π)
                let dEffMM = 2 * sqrt(areaMM2 / .pi)
                let dEffM = Double(dEffMM) / 1000.0  // mm to meters
                print("Effective droplet diameter: \(dEffMM) mm")

                // Parse density or default to water
                let rho = (Double(density) ?? 1000.0)
                let g = 9.81
                // Surface tension: γ = ρ·g·d_eff²
                let gamma = rho * g * dEffM * dEffM
                // Format as mN/m
                return String(format: "Surface Tension: %.1f mN/m", gamma * 1000)

            } catch {
                print("Vision error: \(error.localizedDescription)")
                return nil
            }
        }

        // MARK: - Polygon Area Helper (Shoelace)
        func polygonArea(points: [CGPoint]) -> CGFloat {
            guard points.count > 2 else { return 0 }
            var area: CGFloat = 0
            for i in 0..<points.count {
                let j = (i + 1) % points.count
                area += points[i].x * points[j].y - points[j].x * points[i].y
            }
            return abs(area) / 2
        }

        // MARK: - Flashlight Control
        @objc func toggleFlashlight() {
            guard let device = AVCaptureDevice.default(for: .video), device.hasTorch else { return }
            do {
                try device.lockForConfiguration()
                device.torchMode = (device.torchMode == .on ? .off : .on)
                device.unlockForConfiguration()
            } catch {
                print("Error toggling flashlight: \(error)")
            }
        }

        // MARK: - Tap-to-Focus & Zoom
        @objc func handleTap(_ sender: UITapGestureRecognizer) {
            guard let view = sender.view, let preview = previewLayer else { return }
            let tapPoint = sender.location(in: view)
            let focusPoint = preview.captureDevicePointConverted(fromLayerPoint: tapPoint)
            focusAndZoom(at: focusPoint)
        }

        func focusAndZoom(at focusPoint: CGPoint) {
            guard let device = AVCaptureDevice.default(for: .video) else { return }
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
                // Apply 3x zoom (clamped to max)
                let zoom: CGFloat = min(3.0, device.activeFormat.videoMaxZoomFactor)
                device.videoZoomFactor = zoom
                device.unlockForConfiguration()
            } catch {
                print("Error setting focus/zoom: \(error)")
            }
        }
    } // End Coordinator

    // MARK: Create and update UIViewController
    func makeCoordinator() -> Coordinator { Coordinator(parent: self) }
    func makeUIViewController(
        context: Context
    ) -> UIViewController {
        let vc = UIViewController()

        // Add preview layer for live camera
        let previewLayer = AVCaptureVideoPreviewLayer(session: context.coordinator.session)
        previewLayer.frame = vc.view.bounds
        previewLayer.videoGravity = .resizeAspectFill
        vc.view.layer.addSublayer(previewLayer)
        context.coordinator.previewLayer = previewLayer

        // Gesture recognizer for tap-to-focus/zoom
        let tap = UITapGestureRecognizer(target: context.coordinator, action: #selector(Coordinator.handleTap(_:)))
        vc.view.addGestureRecognizer(tap)

        // Analysis button setup
        let analysisButton = UIButton(type: .system)
        analysisButton.setTitle("Start Analysis", for: .normal)
        analysisButton.backgroundColor = .systemBlue
        analysisButton.setTitleColor(.white, for: .normal)
        analysisButton.layer.cornerRadius = 8
        analysisButton.translatesAutoresizingMaskIntoConstraints = false
        analysisButton.addTarget(
            context.coordinator, action: #selector(Coordinator.capturePhoto), for: .touchUpInside)
        vc.view.addSubview(analysisButton)
        NSLayoutConstraint.activate([
            analysisButton.leadingAnchor.constraint(equalTo: vc.view.leadingAnchor, constant: 20),
            analysisButton.trailingAnchor.constraint(equalTo: vc.view.trailingAnchor, constant: -20),
            analysisButton.bottomAnchor.constraint(equalTo: vc.view.bottomAnchor, constant: -120),
            analysisButton.heightAnchor.constraint(equalToConstant: 40)
        ])

        // Flashlight toggle button
        let flashlightButton = UIButton(type: .system)
        flashlightButton.setTitle("Toggle Flashlight", for: .normal)
        flashlightButton.backgroundColor = .systemYellow
        flashlightButton.setTitleColor(.black, for: .normal)
        flashlightButton.layer.cornerRadius = 8
        flashlightButton.translatesAutoresizingMaskIntoConstraints = false
        flashlightButton.addTarget(
            context.coordinator, action: #selector(Coordinator.toggleFlashlight), for: .touchUpInside)
        vc.view.addSubview(flashlightButton)
        NSLayoutConstraint.activate([
            flashlightButton.leadingAnchor.constraint(equalTo: vc.view.leadingAnchor, constant: 20),
            flashlightButton.trailingAnchor.constraint(equalTo: vc.view.trailingAnchor, constant: -20),
            flashlightButton.bottomAnchor.constraint(equalTo: analysisButton.topAnchor, constant: -10),
            flashlightButton.heightAnchor.constraint(equalToConstant: 40)
        ])

        // Label to display surface tension result
        let label = UILabel()
        label.textAlignment = .center
        label.textColor = .white
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

        // Bind surfaceTension publisher to label text
        context.coordinator.$surfaceTension
            .receive(on: DispatchQueue.main)
            .sink { text in label.text = text }
            .store(in: &context.coordinator.cancellables)

        return vc
    }
    func updateUIViewController(_ uiViewController: UIViewController, context: Context) {}
}

// MARK: - MainView: SwiftUI root providing input fields and embedding ContentView
struct MainView: View {
    @State private var pipetDiameter: String = ""  // User enters diameter
    @State private var density: String = ""        // User enters density

    var body: some View {
        VStack {
            // Input for pipette diameter
            TextField("Enter pipet diameter (mm)", text: $pipetDiameter)
                .textFieldStyle(RoundedBorderTextFieldStyle())
                .keyboardType(.decimalPad)
                .padding()
                .toolbar {
                    ToolbarItemGroup(placement: .keyboard) {
                        Spacer()
                        Button("Done") {
                            UIApplication.shared.sendAction(
                                #selector(UIResponder.resignFirstResponder), to: nil, from: nil, for: nil)
                        }
                    }
                }
            // Input for liquid density
            TextField("Enter density (kg/m³)", text: $density)
                .textFieldStyle(RoundedBorderTextFieldStyle())
                .keyboardType(.decimalPad)
                .padding([.horizontal, .bottom])
                .toolbar {
                    ToolbarItemGroup(placement: .keyboard) {
                        Spacer()
                        Button("Done") {
                            UIApplication.shared.sendAction(
                                #selector(UIResponder.resignFirstResponder), to: nil, from: nil, for: nil)
                        }
                    }
                }
            // Embed the camera analysis view
            ContentView(pipetDiameter: $pipetDiameter, density: $density)
                .edgesIgnoringSafeArea(.all)
        }
    }
}

// MARK: - Preview
struct MainView_Previews: PreviewProvider {
    static var previews: some View {
        MainView()
    }
}

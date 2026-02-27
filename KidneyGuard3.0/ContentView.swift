// MARK: - Framework Imports
import SwiftUI
import UIKit
import AVFoundation
import Vision
import Combine
import CoreImage
import CoreImage.CIFilterBuiltins

// MARK: - ContentView
struct ContentView: UIViewControllerRepresentable {

    @Binding var pipetDiameter: String
    @Binding var density: String

    class Coordinator: NSObject, AVCapturePhotoCaptureDelegate, ObservableObject {

        let session = AVCaptureSession()
        let output = AVCapturePhotoOutput()
        var parent: ContentView

        @Published var surfaceTension: String = "Tap to measure"
        @Published var containerFound: Bool = true

        private var photoCompletion: ((UIImage) -> Void)?
        var cancellables = Set<AnyCancellable>()
        var previewLayer: AVCaptureVideoPreviewLayer?

        init(parent: ContentView) {
            self.parent = parent
            super.init()
            setupCamera()
            checkPermissions()
        }

        // MARK: - Container Validation (manual bounding box + simpler math)
        func validateContainer(from contour: VNContour, imageSize: CGSize) -> Bool {

            let normalized = contour.normalizedPoints
            let nCount = normalized.count
            guard nCount > 200 else { return false }

            // ---- Manual bounding box in normalized (0..1) space ----
            var minX: CGFloat = 1
            var maxX: CGFloat = 0
            var minY: CGFloat = 1
            var maxY: CGFloat = 0

            for p in normalized {
                let x = CGFloat(p.x)
                let y = CGFloat(p.y)
                if x < minX { minX = x }
                if x > maxX { maxX = x }
                if y < minY { minY = y }
                if y > maxY { maxY = y }
            }

            let boxWidth = maxX - minX
            let boxHeight = maxY - minY

            // Must take up enough vertical space
            if boxHeight <= 0.30 { return false }

            // Must be taller than wide (cup/tube-ish)
            if boxHeight <= boxWidth { return false }

            // Convert to pixel points once for remaining checks
            let w = imageSize.width
            let h = imageSize.height

            var points: [CGPoint] = []
            points.reserveCapacity(nCount)

            for p in normalized {
                points.append(CGPoint(x: CGFloat(p.x) * w, y: CGFloat(p.y) * h))
            }

            // Pixel extents
            var pMinX = CGFloat.greatestFiniteMagnitude
            var pMaxX = -CGFloat.greatestFiniteMagnitude
            var pMinY = CGFloat.greatestFiniteMagnitude
            var pMaxY = -CGFloat.greatestFiniteMagnitude

            for pt in points {
                if pt.x < pMinX { pMinX = pt.x }
                if pt.x > pMaxX { pMaxX = pt.x }
                if pt.y < pMinY { pMinY = pt.y }
                if pt.y > pMaxY { pMaxY = pt.y }
            }

            if !(pMinX.isFinite && pMaxX.isFinite && pMinY.isFinite && pMaxY.isFinite) { return false }

            let midX = (pMinX + pMaxX) * 0.5
            let imageMidX = w * 0.5

            // Must be roughly centered
            if abs(midX - imageMidX) >= w * 0.25 { return false }

            // Check vertical width consistency (cylindrical-ish)
            let heightPx = pMaxY - pMinY
            if heightPx <= 0 { return false }

            let slices = 6
            let yTolerance = heightPx * 0.03

            var widths: [CGFloat] = []
            widths.reserveCapacity(slices)

            for i in 0..<slices {
                let t = CGFloat(i) / CGFloat(max(slices - 1, 1))
                let yLevel = pMinY + t * heightPx

                var sliceMinX = CGFloat.greatestFiniteMagnitude
                var sliceMaxX = -CGFloat.greatestFiniteMagnitude
                var sliceCount = 0

                for pt in points {
                    if abs(pt.y - yLevel) < yTolerance {
                        sliceCount += 1
                        if pt.x < sliceMinX { sliceMinX = pt.x }
                        if pt.x > sliceMaxX { sliceMaxX = pt.x }
                    }
                }

                if sliceCount > 5, sliceMinX.isFinite, sliceMaxX.isFinite {
                    widths.append(sliceMaxX - sliceMinX)
                }
            }

            if widths.count < 3 { return false }

            var sum: CGFloat = 0
            for v in widths { sum += v }
            let avgWidth = sum / CGFloat(widths.count)
            if avgWidth <= 0 { return false }

            var devSum: CGFloat = 0
            for v in widths { devSum += abs(v - avgWidth) }
            let variance = devSum / CGFloat(widths.count)

            if variance >= avgWidth * 0.25 { return false }

            // Left-right symmetry check
            var leftCount = 0
            var rightCount = 0

            for pt in points {
                if pt.x < midX { leftCount += 1 }
                else if pt.x > midX { rightCount += 1 }
            }

            let diff = abs(leftCount - rightCount)
            if CGFloat(diff) >= CGFloat(points.count) * 0.20 { return false }

            return true
        }

        // MARK: Camera Setup
        func setupCamera() {
            session.sessionPreset = .high

            guard let device = AVCaptureDevice.default(
                .builtInWideAngleCamera,
                for: .video,
                position: .back
            ),
            let input = try? AVCaptureDeviceInput(device: device)
            else { return }

            do {
                try device.lockForConfiguration()

                if device.isFocusModeSupported(.continuousAutoFocus) {
                    device.focusMode = .continuousAutoFocus
                    device.focusPointOfInterest = CGPoint(x: 0.5, y: 0.5)
                }

                if device.isExposureModeSupported(.continuousAutoExposure) {
                    device.exposureMode = .continuousAutoExposure
                    device.exposurePointOfInterest = CGPoint(x: 0.5, y: 0.5)
                }

                device.unlockForConfiguration()
            } catch { }

            if session.canAddInput(input) { session.addInput(input) }
            if session.canAddOutput(output) { session.addOutput(output) }
        }

        // MARK: Permissions
        func checkPermissions() {
            switch AVCaptureDevice.authorizationStatus(for: .video) {
            case .authorized:
                DispatchQueue.global().async { self.session.startRunning() }

            case .notDetermined:
                AVCaptureDevice.requestAccess(for: .video) { granted in
                    if granted {
                        DispatchQueue.global().async { self.session.startRunning() }
                    } else {
                        DispatchQueue.main.async {
                            self.surfaceTension = "Camera access denied"
                        }
                    }
                }

            default:
                surfaceTension = "Camera access denied"
            }
        }

        // MARK: Capture Photo
        @objc func capturePhoto() {
            let settings = AVCapturePhotoSettings()

            photoCompletion = { image in
                DispatchQueue.global(qos: .userInitiated).async {
                    let result = self.calculateSurfaceTension(
                        from: image,
                        pipetDiameter: self.parent.pipetDiameter,
                        density: self.parent.density
                    )

                    DispatchQueue.main.async {
                        self.surfaceTension = result ?? "Measurement failed"
                    }
                }
            }

            output.capturePhoto(with: settings, delegate: self)
        }

        func photoOutput(
            _ output: AVCapturePhotoOutput,
            didFinishProcessingPhoto photo: AVCapturePhoto,
            error: Error?
        ) {
            guard let data = photo.fileDataRepresentation(),
                  let image = UIImage(data: data)
            else { return }

            photoCompletion?(image)
        }

        // MARK: - Pendant Drop Surface Tension Calculation
        func calculateSurfaceTension(
            from image: UIImage,
            pipetDiameter: String,
            density: String
        ) -> String? {

            guard let cgImage = image.cgImage else { return nil }

            // ---- SIMPLE SEGMENTATION (Threshold Based) ----
            let context = CIContext()
            let inputCI = CIImage(cgImage: cgImage)

            // Convert to grayscale
            let grayscale = CIFilter.colorControls()
            grayscale.inputImage = inputCI
            grayscale.saturation = 0
            grayscale.contrast = 1.5
            grayscale.brightness = 0

            guard let grayOutput = grayscale.outputImage else { return nil }

            // Apply threshold using color matrix trick
            let thresholdFilter = CIFilter.colorMatrix()
            thresholdFilter.inputImage = grayOutput

            // Adjust threshold level (tune if needed)
            let threshold: CGFloat = 0.6

            thresholdFilter.rVector = CIVector(x: 10, y: 0, z: 0, w: -threshold * 10)
            thresholdFilter.gVector = CIVector(x: 0, y: 10, z: 0, w: -threshold * 10)
            thresholdFilter.bVector = CIVector(x: 0, y: 0, z: 10, w: -threshold * 10)
            thresholdFilter.aVector = CIVector(x: 0, y: 0, z: 0, w: 1)

            guard let thresholded = thresholdFilter.outputImage,
                  let processedCG = context.createCGImage(thresholded, from: thresholded.extent)
            else { return nil }

            // Now run Vision on CLEAN binary image
            let request = VNDetectContoursRequest()
            request.contrastAdjustment = 1.0
            request.detectDarkOnLight = true

            let handler = VNImageRequestHandler(cgImage: processedCG, options: [:])

            do {
                try handler.perform([request])

                guard let obs = request.results?.first as? VNContoursObservation else {
                    DispatchQueue.main.async { self.containerFound = false }
                    return nil
                }

                guard let contour = obs.topLevelContours.max(by: { $0.pointCount < $1.pointCount }) else {
                    DispatchQueue.main.async { self.containerFound = false }
                    return nil
                }

                let isValidContainer = validateContainer(
                    from: contour,
                    imageSize: CGSize(width: cgImage.width, height: cgImage.height)
                )

                DispatchQueue.main.async {
                    self.containerFound = isValidContainer
                }

                guard isValidContainer else { return nil }

                let pixelPoints = contour.normalizedPoints.map {
                    CGPoint(
                        x: CGFloat($0.x) * CGFloat(cgImage.width),
                        y: CGFloat($0.y) * CGFloat(cgImage.height)
                    )
                }

                // ---- STRAW WIDTH DETECTION (for proper scaling) ----
                guard let d = Double(pipetDiameter), d > 0 else { return nil }

                // Find top of droplet (apex)
                guard let minY = pixelPoints.map({ $0.y }).min() else { return nil }

                // Define vertical band where straw should be located
                let strawBandHeight = CGFloat(cgImage.height) * 0.05

                let strawPoints = pixelPoints.filter {
                    $0.y < (minY + strawBandHeight)
                }

                // Ensure we found enough points
                guard strawPoints.count > 5 else { return nil }

                // Measure straw pixel width
                let strawXs = strawPoints.map { $0.x }

                guard let strawMinX = strawXs.min(),
                      let strawMaxX = strawXs.max()
                else { return nil }

                let strawPixelWidth = strawMaxX - strawMinX
                guard strawPixelWidth > 0 else { return nil }

                let scaleMMPerPixel = CGFloat(d) / strawPixelWidth

                // ---- ROBUST APEX SELECTION ----
                guard let minY2 = pixelPoints.map({ $0.y }).min() else { return nil }
                let apexBandHeight = CGFloat(cgImage.height) * 0.025

                let apexPoints = pixelPoints.filter {
                    $0.y >= minY2 && $0.y <= (minY2 + apexBandHeight)
                }

                guard apexPoints.count > 5 else { return nil }

                guard let radiusPixels = fitCircleRadius(points: apexPoints) else {
                    return nil
                }

                let radiusMM = radiusPixels * scaleMMPerPixel
                let radiusMeters = Double(radiusMM) / 1000.0

                // ---- PHYSICS (Pendant Drop Approximation) ----
                let rho = Double(density) ?? 1000.0
                let g = 9.81

                let gamma = (rho * g * radiusMeters * radiusMeters) / 2.0
                return String(format: "Surface Tension: %.2f mN/m", gamma * 1000)

            } catch {
                return nil
            }
        }

        func fitCircleRadius(points: [CGPoint]) -> CGFloat? {
            guard points.count > 3 else { return nil }

            let n = CGFloat(points.count)

            var sumX: CGFloat = 0
            var sumY: CGFloat = 0
            var sumX2: CGFloat = 0
            var sumY2: CGFloat = 0
            var sumXY: CGFloat = 0
            var sumX3: CGFloat = 0
            var sumY3: CGFloat = 0
            var sumX1Y2: CGFloat = 0
            var sumX2Y1: CGFloat = 0

            for p in points {
                let x = p.x
                let y = p.y
                let x2 = x * x
                let y2 = y * y

                sumX += x
                sumY += y
                sumX2 += x2
                sumY2 += y2
                sumXY += x * y
                sumX3 += x2 * x
                sumY3 += y2 * y
                sumX1Y2 += x * y2
                sumX2Y1 += x2 * y
            }

            let C = n * sumX2 - sumX * sumX
            let D = n * sumXY - sumX * sumY
            let E = n * sumY2 - sumY * sumY
            let G = 0.5 * (n * sumX3 + n * sumX1Y2 - (sumX2 + sumY2) * sumX)
            let H = 0.5 * (n * sumY3 + n * sumX2Y1 - (sumX2 + sumY2) * sumY)

            let denominator = C * E - D * D
            guard abs(denominator) > 1e-6 else { return nil }

            let centerX = (G * E - D * H) / denominator
            let centerY = (C * H - D * G) / denominator

            let radius = sqrt(
                (centerX - sumX / n) * (centerX - sumX / n) +
                (centerY - sumY / n) * (centerY - sumY / n) +
                (sumX2 + sumY2) / n
            )

            return radius
        }

        // MARK: Flashlight
        @objc func toggleFlashlight() {
            guard let device = AVCaptureDevice.default(for: .video),
                  device.hasTorch else { return }
            do {
                try device.lockForConfiguration()
                device.torchMode = (device.torchMode == .on ? .off : .on)
                device.unlockForConfiguration()
            } catch { }
        }

        // MARK: Tap to Focus & Zoom
        @objc func handleTap(_ sender: UITapGestureRecognizer) {
            guard let view = sender.view,
                  let preview = previewLayer else { return }
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

                let zoom: CGFloat = min(3.0, device.activeFormat.videoMaxZoomFactor)
                device.videoZoomFactor = zoom

                device.unlockForConfiguration()
            } catch { }
        }
    }

    // MARK: UIViewControllerRepresentable
    func makeCoordinator() -> Coordinator { Coordinator(parent: self) }

    func makeUIViewController(context: Context) -> UIViewController {
        let vc = UIViewController()

        let previewLayer = AVCaptureVideoPreviewLayer(session: context.coordinator.session)
        previewLayer.frame = vc.view.bounds
        previewLayer.videoGravity = .resizeAspectFill
        vc.view.layer.addSublayer(previewLayer)
        context.coordinator.previewLayer = previewLayer

        let tap = UITapGestureRecognizer(
            target: context.coordinator,
            action: #selector(Coordinator.handleTap(_:))
        )
        vc.view.addGestureRecognizer(tap)

        let analysisButton = UIButton(type: .system)
        analysisButton.setTitle("Start Analysis", for: .normal)
        analysisButton.backgroundColor = .systemBlue
        analysisButton.setTitleColor(.white, for: .normal)
        analysisButton.layer.cornerRadius = 8
        analysisButton.translatesAutoresizingMaskIntoConstraints = false
        analysisButton.addTarget(
            context.coordinator,
            action: #selector(Coordinator.capturePhoto),
            for: .touchUpInside
        )

        vc.view.addSubview(analysisButton)

        NSLayoutConstraint.activate([
            analysisButton.leadingAnchor.constraint(equalTo: vc.view.leadingAnchor, constant: 20),
            analysisButton.trailingAnchor.constraint(equalTo: vc.view.trailingAnchor, constant: -20),
            analysisButton.bottomAnchor.constraint(equalTo: vc.view.bottomAnchor, constant: -120),
            analysisButton.heightAnchor.constraint(equalToConstant: 40)
        ])

        let flashlightButton = UIButton(type: .system)
        flashlightButton.setTitle("Toggle Flashlight", for: .normal)
        flashlightButton.backgroundColor = .systemYellow
        flashlightButton.setTitleColor(.black, for: .normal)
        flashlightButton.layer.cornerRadius = 8
        flashlightButton.translatesAutoresizingMaskIntoConstraints = false
        flashlightButton.addTarget(
            context.coordinator,
            action: #selector(Coordinator.toggleFlashlight),
            for: .touchUpInside
        )

        vc.view.addSubview(flashlightButton)

        NSLayoutConstraint.activate([
            flashlightButton.leadingAnchor.constraint(equalTo: vc.view.leadingAnchor, constant: 20),
            flashlightButton.trailingAnchor.constraint(equalTo: vc.view.trailingAnchor, constant: -20),
            flashlightButton.bottomAnchor.constraint(equalTo: analysisButton.topAnchor, constant: -10),
            flashlightButton.heightAnchor.constraint(equalToConstant: 40)
        ])

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

        context.coordinator.$surfaceTension
            .receive(on: DispatchQueue.main)
            .sink { text in label.text = text }
            .store(in: &context.coordinator.cancellables)

        return vc
    }

    func updateUIViewController(_ uiViewController: UIViewController, context: Context) { }
}

// MARK: - MainView
struct MainView: View {
    @State private var pipetDiameter: String = ""
    @State private var density: String = ""

    @FocusState private var focusedField: Field?
    enum Field { case pipet, density }

    var body: some View {
        VStack {
            TextField("Enter pipet diameter (mm)", text: $pipetDiameter)
                .textFieldStyle(RoundedBorderTextFieldStyle())
                .keyboardType(.decimalPad)
                .focused($focusedField, equals: .pipet)
                .padding()

            TextField("Enter density (kg/m³)", text: $density)
                .textFieldStyle(RoundedBorderTextFieldStyle())
                .keyboardType(.decimalPad)
                .focused($focusedField, equals: .density)
                .padding([.horizontal, .bottom])

            ContentView(pipetDiameter: $pipetDiameter,
                        density: $density)
                .edgesIgnoringSafeArea(.all)
        }
        .toolbar {
            ToolbarItemGroup(placement: .keyboard) {
                Spacer()
                Button("Done") { focusedField = nil }
            }
        }
        .onTapGesture {
            focusedField = nil
        }
    }
}

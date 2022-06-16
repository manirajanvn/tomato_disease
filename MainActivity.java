package com.plant.tomato2;
import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.Camera;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import androidx.lifecycle.LifecycleOwner;

import android.annotation.SuppressLint;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.util.Log;
import android.util.Size;
import android.widget.TextView;

import com.google.common.util.concurrent.ListenableFuture;

import org.pytorch.IValue;
import org.pytorch.LiteModuleLoader;
import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executor;
import java.util.concurrent.Executors;

public class MainActivity extends AppCompatActivity {

    private ListenableFuture<ProcessCameraProvider> camareProviderFuture;

    PreviewView previewView;
    TextView textView;

    private int REQUEST_CODE_PERMISSION = 101;
    private final String[] REQUIRED_PERMISSIONS = new String[] {"android.permission.CAMERA"};

    List<String> tomato_classes;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        previewView = findViewById(R.id.cameraView);
        textView = findViewById(R.id.resule_text);

        if(checkPermission()){
            ActivityCompat.requestPermissions(this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSION);
        }
        tomato_classes = loadClasses("image_classes.txt");
        loadTorchModel("tomato_mobile.ptl");
        camareProviderFuture = ProcessCameraProvider.getInstance(this);
        camareProviderFuture.addListener(()  -> {
            try{
                ProcessCameraProvider cameraProvider = camareProviderFuture.get();
                startCamera(cameraProvider);
            }catch(ExecutionException | InterruptedException e){
                e.printStackTrace();
            }
        }, ContextCompat.getMainExecutor(this));
    }

    private boolean checkPermission(){
        for (String permission : REQUIRED_PERMISSIONS){
            if(ContextCompat.checkSelfPermission(this, permission) != PackageManager.PERMISSION_GRANTED){
                return false;
            }
        }
        return  true;
    }

    Executor executor = Executors.newSingleThreadExecutor();
    void startCamera(ProcessCameraProvider cameraProvider){
        Preview preview = new Preview.Builder().build();
        CameraSelector cameraSelector = new CameraSelector.Builder().requireLensFacing(CameraSelector.LENS_FACING_BACK).build();
        preview.setSurfaceProvider(previewView.getSurfaceProvider());
        ImageAnalysis imageAnalysis = new ImageAnalysis.Builder().setTargetResolution(new Size(224, 224))
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST).build();
        imageAnalysis.setAnalyzer(executor, new ImageAnalysis.Analyzer() {
            @Override
            public void analyze(@NonNull ImageProxy image) {
                int rotation = image.getImageInfo().getRotationDegrees();
                Log.v("Torch", "Rotate: " + rotation);
                analyzeImage(image, rotation);
                image.close();
            }
        });
        Camera camera = cameraProvider.bindToLifecycle((LifecycleOwner) this, cameraSelector, preview, imageAnalysis);
    }

    Module module;

    void loadTorchModel(String fileName){
        File modelFIle = new File(this.getFilesDir(),fileName);
        try{
            if(!modelFIle.exists()){
                InputStream inputStream = getAssets().open(fileName);
                FileOutputStream outputStream = new FileOutputStream(modelFIle);
                byte[] buffer = new byte[2048];
                int byteRead = -1;
                while((byteRead = inputStream.read(buffer)) != -1){
                    outputStream.write(buffer, 0, byteRead);
                }
                inputStream.close();
                outputStream.close();
            }
            module = LiteModuleLoader.load(modelFIle.getAbsolutePath());
        }catch(Exception e){
            e.printStackTrace();
        }
    }

    void analyzeImage(ImageProxy image, int rotation){
        @SuppressLint("UnsafeOptInUsageError") Tensor inputTensor = TensorImageUtils.imageYUV420CenterCropToFloat32Tensor(image.getImage(),
                rotation, 224, 224, TensorImageUtils.TORCHVISION_NORM_MEAN_RGB,
                TensorImageUtils.TORCHVISION_NORM_STD_RGB);
        Tensor outputTensor = module.forward(IValue.from(inputTensor)).toTensor();
        float[] scores = outputTensor.getDataAsFloatArray();
        float maxScore = -Float.MAX_VALUE;
        int maxScoreIdx = -1;
        for(int i=0; i<scores.length; i++ ){
            if(scores[i] > maxScore){
                maxScore = scores[i];
                maxScoreIdx = i;
            }
        }

        String classResult = tomato_classes.get(maxScoreIdx);
        Log.v("Torch", "Detected: " + classResult);

        runOnUiThread(new Runnable() {
            @Override
            public void run() {
                textView.setText(classResult);
            }
        });
    }


    List<String> loadClasses(String fileName) {
        List<String> classes = new ArrayList<String>();
        try {
            BufferedReader br = new BufferedReader(new InputStreamReader(getAssets().open(fileName)));
            String line;
            while((line = br.readLine()) != null){
                classes.add(line);
            }
        }catch(Exception e){
            e.printStackTrace();
        }
        return classes;
    }

}
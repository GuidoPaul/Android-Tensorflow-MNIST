package com.guido.mnist;

import android.graphics.PointF;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.MotionEvent;
import android.view.View;
import android.widget.TextView;

import com.guido.mnist.classifier.Classifier;
import com.guido.mnist.classifier.TensorFlowImageClassifier;
import com.guido.mnist.view.DrawModel;
import com.guido.mnist.view.DrawView;

import java.util.List;
import java.util.concurrent.Executor;
import java.util.concurrent.Executors;

public class MainActivity extends AppCompatActivity implements View.OnTouchListener {

    private static final String TAG = "MainActivity";

    private static final int PIXEL_WIDTH = 28;
    private static final int PIXEL_HEIGHT = 28;
    private static final String MODEL_FILE = "file:///android_asset/mnist_graph_frozen.pb";
    private static final String LABEL_FILE = "file:///android_asset/graph_label_strings.txt";
    private static final int INPUT_SIZE = 28;
    private static final String INPUT_NAME = "inputs/inputs";
    private static final String OUTPUT_NAME = "readout/output";

    private Classifier classifier;

    private DrawModel mModel;
    private DrawView mDrawView;
    private TextView mResultText;

    private float mLastX;
    private float mLastY;
    private PointF mTmpPoint = new PointF();

    private Executor executor = Executors.newSingleThreadExecutor();

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        mModel = new DrawModel(PIXEL_WIDTH, PIXEL_HEIGHT);
        mDrawView = (DrawView) findViewById(R.id.view_draw);
        mDrawView.setModel(mModel);
        mDrawView.setOnTouchListener(this);

        mResultText = (TextView) findViewById(R.id.textResult);

        View buttonClean = findViewById(R.id.buttonClean);
        buttonClean.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                onCleanClicked();
            }
        });

        View buttonDetect = findViewById(R.id.buttonDetect);
        buttonDetect.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                onDetectClicked();
            }
        });

        initTensorFlowAndLoadModel();
    }

    private void initTensorFlowAndLoadModel() {
        executor.execute(new Runnable() {
            @Override
            public void run() {
                try {
                    classifier = TensorFlowImageClassifier.create(
                            getAssets(), MODEL_FILE, LABEL_FILE, INPUT_SIZE, INPUT_NAME,
                            OUTPUT_NAME);
                    Log.d(TAG, "Load Success");
                } catch (Exception e) {
                    throw new RuntimeException("Error initializing Tensorflow!", e);
                }
            }
        });
    }


    private void onDetectClicked() {
        float[] pixels = mDrawView.getPixelData();

        List<Classifier.Recognition> results = classifier.recognizeImage(pixels);

        if (results.size() > 0) {
            String value = results.get(0).getTitle();
            mResultText.setText(value);
        }

    }

    private void onCleanClicked() {
        mModel.clear();
        mDrawView.reset();
        mDrawView.invalidate();

        mResultText.setText("");
    }

    @Override
    public boolean onTouch(View v, MotionEvent event) {
        int action = event.getAction() & MotionEvent.ACTION_MASK;

        if (action == MotionEvent.ACTION_DOWN) {
            processTouchDown(event);
            return true;
        } else if (action == MotionEvent.ACTION_MOVE) {
            processTouchMove(event);
            return true;
        } else if (action == MotionEvent.ACTION_UP) {
            processTouchUp();
            return true;
        }

        return false;
    }

    private void processTouchDown(MotionEvent event) {
        mLastX = event.getX();
        mLastY = event.getY();
        mDrawView.calcPos(mLastX, mLastY, mTmpPoint);
        float lastConvX = mTmpPoint.x;
        float lastConvY = mTmpPoint.y;
        mModel.startLine(lastConvX, lastConvY);
    }

    private void processTouchMove(MotionEvent event) {
        float x = event.getX();
        float y = event.getY();

        mDrawView.calcPos(x, y, mTmpPoint);
        float newConvX = mTmpPoint.x;
        float newConvY = mTmpPoint.y;
        mModel.addLineElem(newConvX, newConvY);

        mLastX = x;
        mLastY = y;
        mDrawView.invalidate();
    }

    private void processTouchUp() {
        mModel.endLine();
    }

    @Override
    protected void onResume() {
        mDrawView.onResume();
        super.onResume();
    }

    @Override
    protected void onPause() {
        mDrawView.onPause();
        super.onPause();
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        executor.execute(new Runnable() {
            @Override
            public void run() {
                classifier.close();
            }
        });
    }
}
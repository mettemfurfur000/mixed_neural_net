#include "../include/dataset.h"
#include "../include/funcs.h"
#include "../include/general.h"
#include "../include/io.h"
#include "../include/nn.h"

#include <raylib.h>

#include <stdio.h>
#include <string.h>
#include <time.h>

typedef struct
{
    Rectangle bounds;
    const char *label;
    bool hovered;
    bool pressed;
} button_t;

typedef struct
{
    neural_net *net;
    neural_func_list funcs;
    layer layers_array[16];
    // dataset *test_dataset;
    train_test_split *mnist_split;
    float *draw_buffer;
    prediction_result *prediction;
    bool has_network;
    int connection_count;
    int *func_counts;
    float *func_percentages;
} app_state_t;

float avg_fitness(neural_net *net, dataset *test, float coverage)
{
    float fitness_sum = 0;
    float fitness_avg = 0;

    for (u32 i = 0; i < test->num_samples; i++)
    {
        if (rand() / (float)RAND_MAX > coverage)
            continue;
        float cur_fit = neural_net_fitness(net, test->samples[i].features, test->samples[i].label);

        fitness_sum += cur_fit;
    }

    fitness_avg = fitness_sum / (test->num_samples * coverage);

    return fitness_avg;
}

void count_network_stats(app_state_t *app)
{
    if (!app->net || !app->has_network)
        return;

    // Count connections
    app->connection_count = 0;
    for (int i = 0; i < app->net->size - 1; i++)
    {
        layer *l = &app->net->layers[i];
        layer *next = &app->net->layers[i + 1];
        app->connection_count += l->size * next->size;
    }

    // Count activation functions
    if (!app->func_counts)
        app->func_counts = calloc(5, sizeof(int));
    if (!app->func_percentages)
        app->func_percentages = calloc(5, sizeof(float));

    if (!app->func_counts || !app->func_percentages)
        return; // Allocation failed

    memset(app->func_counts, 0, 5 * sizeof(int));

    int total_neurons = 0;
    for (int i = 0; i < app->net->size; i++)
    {
        layer *l = &app->net->layers[i];
        for (int j = 0; j < l->size; j++)
        {
            total_neurons++;
            if (l->neurons[j].activation.f == line)
                app->func_counts[FUNC_LINE]++;
            else if (l->neurons[j].activation.f == step)
                app->func_counts[FUNC_STEP]++;
            else if (l->neurons[j].activation.f == relu)
                app->func_counts[FUNC_RELU]++;
            else if (l->neurons[j].activation.f == sigmoid)
                app->func_counts[FUNC_SIGMOID]++;
            else if (l->neurons[j].activation.f == softsign)
                app->func_counts[FUNC_SOFTSIGN]++;
        }
    }

    for (int i = 0; i < 5; i++)
    {
        app->func_percentages[i] = total_neurons > 0 ? (app->func_counts[i] * 100.0f / total_neurons) : 0.0f;
    }
}

void generate_network(app_state_t *app)
{
    srand((unsigned int)time(NULL));

    init_neural_func_list(&app->funcs);

    u32 i = 0;
    create_layer(&app->layers_array[i++], 28 * 28, app->funcs.funcs[FUNC_SIGMOID]);
    create_layer(&app->layers_array[i++], 128, app->funcs.funcs[FUNC_SIGMOID]);
    create_layer(&app->layers_array[i++], 32, app->funcs.funcs[FUNC_SIGMOID]);
    // create_layer(&app->layers_array[i++], 16, app->funcs.funcs[FUNC_SIGMOID]);
    create_layer(&app->layers_array[i++], 10, app->funcs.funcs[FUNC_SIGMOID]); // 10 outputs for digits 0-9

    app->net = create_neural_net(i, app->layers_array);
    app->has_network = true;

    printf("Network generated with 10 output neurons for digits 0-9\n");

    count_network_stats(app);
}

void load_mnist_dataset(app_state_t *app)
{
    if (!app->net || !app->has_network)
        return;

    printf("Loading dataset...\n");

    dataset *data = load_csv_dataset_with_label_pos("mnist.csv", 28 * 28, 1, true, 1);
    if (!data)
    {
        printf("Failed to load dataset\n");
        return;
    }

    normalize_dataset(data, 0.0f, 1.0f);

    if (app->mnist_split)
        free_train_test_split(app->mnist_split);

    app->mnist_split = calloc(1, sizeof(train_test_split));

    assert(app->mnist_split);

    *(app->mnist_split) = split_dataset(data, 0.8f);

    free_dataset(data);
    printf("Dataset loaded. Training samples: %d, Test samples: %d\n", app->mnist_split->train->num_samples,
           app->mnist_split->test->num_samples);
}

void train_network(app_state_t *app)
{
    if (!app->net || !app->has_network)
        return;

    if (!app->mnist_split)
    {
        load_mnist_dataset(app);
    }

    train_test_split split = *(app->mnist_split);

    // Expand label size from 1 to 10 to accommodate one-hot encoding
    if (!expand_dataset_labels(split.train, 10))
    {
        printf("Failed to expand training labels\n");
        free_train_test_split(&split);
        return;
    }

    if (!expand_dataset_labels(split.test, 10))
    {
        printf("Failed to expand test labels\n");
        free_train_test_split(&split);
        return;
    }

    // Convert labels to one-hot encoding for all samples
    for (u32 i = 0; i < split.train->num_samples; i++)
    {
        float original_label = split.train->samples[i].label[0];

        label_to_one_hot(original_label, split.train->samples[i].label, 10);
    }

    for (u32 i = 0; i < split.test->num_samples; i++)
    {
        float original_label = split.test->samples[i].label[0];

        label_to_one_hot(original_label, split.test->samples[i].label, 10);
    }

    u32 epochs = 100000;
    u32 mutation_period = 500;
    u32 max_steps_without_improvement = 15;
    u32 total_steps_without_improvement = 0;

    float mutation_chance = 0.99f;
    float mutation_decay = 0.95f;

    float last_fitness = 99999.0f;
    float best_fitness = 99999.0f;

    printf("Training for %d epochs...\n", epochs);
    for (int i = 0; i < epochs; i++)
    {
        data_sample *s = dataset_random_sample(split.train);
        train_neural_net(app->net, s->features, s->label, 0.01f);

        if (i % mutation_period == 0)
        {
            float fit = avg_fitness(app->net, split.train, 0.01f);

            if (fit < 0.001f)
                break;
            if (fit >= last_fitness)
            {
                total_steps_without_improvement++;
            }
            else
            {
                printf("Fitness %.5f at epoch %d...\n", fit, i);
            }

            // Save checkpoint if we achieved better fitness
            if (fit < best_fitness)
            {
                best_fitness = fit;
                if (save_neural_net("mnist_best.nnw", app->net, &app->funcs) == SUCCESS)
                {
                    printf("  Checkpoint saved (best fitness: %.5f)\n", best_fitness);
                }
            }

            if (total_steps_without_improvement >= max_steps_without_improvement)
            {
                // printf("No improvement for %d epochs, stopping training\n", total_epochs_without_improvement);
                // break;
                total_steps_without_improvement = 0;
                printf("No improvement, mutating\n");
                find_better_function(app->net, &app->funcs, s->features, s->label, mutation_chance);
                mutation_chance *= mutation_decay;
            }

            last_fitness = fit;
        }
    }

    printf("Training complete\n");

    // load best network

    neural_net *best_net = load_neural_net("mnist_best.nnw", &app->funcs);
    if (best_net)
    {
        free_neural_net(app->net);
        app->net = best_net;
        printf("Best network loaded\n");
    }

    count_network_stats(app);
}

void load_network(app_state_t *app, const char *filename)
{
    if (app->net && app->has_network)
    {
        free_neural_net(app->net);
        app->has_network = false;
    }

    app->net = load_neural_net(filename, &app->funcs);
    if (app->net)
    {
        app->has_network = true;
        printf("Network loaded from %s\n", filename);
        count_network_stats(app);
    }
    else
    {
        printf("Failed to load network from %s\n", filename);
    }
}

void save_network(app_state_t *app, const char *filename)
{
    if (!app->net || !app->has_network)
    {
        printf("No network to save\n");
        return;
    }

    if (save_neural_net(filename, app->net, &app->funcs) == SUCCESS)
    {
        printf("Network saved to %s\n", filename);
    }
    else
    {
        printf("Failed to save network\n");
    }
}

void draw_button(button_t *btn, bool mouse_over, bool mouse_pressed)
{
    btn->hovered = mouse_over && CheckCollisionPointRec(GetMousePosition(), btn->bounds);
    btn->pressed = btn->hovered && mouse_pressed && IsMouseButtonPressed(MOUSE_LEFT_BUTTON);

    Color btn_color = btn->hovered ? DARKGRAY : LIGHTGRAY;
    if (btn->pressed)
        btn_color = GRAY;

    DrawRectangleRec(btn->bounds, btn_color);
    DrawRectangleLinesEx(btn->bounds, 2, BLACK);

    int text_width = MeasureText(btn->label, 20);
    int text_x = btn->bounds.x + (btn->bounds.width - text_width) / 2;
    int text_y = btn->bounds.y + (btn->bounds.height - 20) / 2;
    DrawText(btn->label, text_x, text_y, 20, BLACK);
}

void draw_drawing_area(Rectangle area, float *buffer, int width, int height, float scale)
{
    // Draw border
    DrawRectangleLinesEx(area, 3, BLACK);

    // Draw grid scaled up
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            float val = buffer[y * width + x];
            Color c = {(u8)(val * 255), (u8)(val * 255), (u8)(val * 255), 255};

            float x_pos = area.x + x * scale;
            float y_pos = area.y + y * scale;

            DrawRectangle(x_pos, y_pos, scale, scale, c);
        }
    }
}

void handle_drawing(Rectangle area, float *buffer, int width, int height, float scale)
{
    if (!CheckCollisionPointRec(GetMousePosition(), area))
        return;

    if (IsMouseButtonDown(MOUSE_LEFT_BUTTON))
    {
        Vector2 mouse = GetMousePosition();
        int grid_x = (int)((mouse.x - area.x) / scale);
        int grid_y = (int)((mouse.y - area.y) / scale);

        if (grid_x >= 0 && grid_x < width && grid_y >= 0 && grid_y < height)
        {
            buffer[grid_y * width + grid_x] = 1.0f;
            // Draw to neighbors too for smoother drawing
            for (int dy = -1; dy <= 1; dy++)
            {
                for (int dx = -1; dx <= 1; dx++)
                {
                    int nx = grid_x + dx;
                    int ny = grid_y + dy;
                    if (nx >= 0 && nx < width && ny >= 0 && ny < height)
                    {
                        buffer[ny * width + nx] = CLAMP(buffer[ny * width + nx] + 0.3f, 0.0f, 1.0f);
                    }
                }
            }
        }
    }
}

void reset_drawing(float *buffer, int width, int height)
{
    memset(buffer, 0, width * height * sizeof(float));
}

void load_random_sample(app_state_t *app, float *buffer, int width, int height)
{
    if (!app->mnist_split)
    {
        load_mnist_dataset(app);
    }

    data_sample *sample = dataset_random_sample(app->mnist_split->test);
    if (sample)
    {
        memcpy(buffer, sample->features, width * height * sizeof(float));
        // Find which digit this sample represents (find the index with value 1.0 in one-hot encoding)
        int digit = 0;
        for (int i = 0; i < sample->label_size; i++)
        {
            if (sample->label[i] > 0.5f)
            {
                digit = i;
                break;
            }
        }
        printf("Loaded random sample with digit: %d\n", digit);
    }
}

void predict_digit(app_state_t *app, float *buffer)
{
    if (!app->net || !app->has_network)
    {
        if (app->prediction)
        {
            free_prediction_result(app->prediction);
            app->prediction = NULL;
        }
        return;
    }

    forward_pass(app->net, buffer);
    layer *output_layer = &app->net->layers[app->net->size - 1];

    // Free old prediction if it exists
    if (app->prediction)
    {
        free_prediction_result(app->prediction);
    }

    // Create new prediction from output layer
    float *outputs = (float *)malloc(output_layer->size * sizeof(float));
    if (outputs)
    {
        for (int i = 0; i < output_layer->size; i++)
        {
            outputs[i] = output_layer->neurons[i].output;
        }
        app->prediction = network_outputs_to_prediction(outputs, output_layer->size);
        free(outputs);
    }
}

int main(int argc, char *argv[])
{
    const int screenWidth = 1000;
    const int screenHeight = 600;

    InitWindow(screenWidth, screenHeight, "MNIST Neural Network Solver");
    SetTargetFPS(60);

    app_state_t app = {0};
    app.draw_buffer = calloc(28 * 28, sizeof(float));

    // Create buttons
    button_t btn_generate = {
        {10, 10, 150, 40},
        "Generate Network", false, false
    };
    button_t btn_train = {
        {170, 10, 150, 40},
        "Train Network", false, false
    };
    button_t btn_save = {
        {330, 10, 150, 40},
        "Save Network", false, false
    };
    button_t btn_load = {
        {490, 10, 150, 40},
        "Load Network", false, false
    };
    button_t btn_reset = {
        {650, 10, 150, 40},
        "Reset Drawing", false, false
    };
    button_t btn_random = {
        {810, 10, 150, 40},
        "Random Sample", false, false
    };

    // Drawing area: 28x28 grid scaled to 280x280 pixels
    float draw_scale = 10.0f;
    Rectangle draw_area = {50, 100, 280, 280};

    printf("MNIST Neural Network Solver loaded\n");
    printf("Press 'G' to generate, 'T' to train, 'S' to save, 'L' to load\n");

    while (!WindowShouldClose())
    {
        // Update
        bool mouse_pressed = IsMouseButtonDown(MOUSE_LEFT_BUTTON);

        draw_button(&btn_generate, true, mouse_pressed);
        if (btn_generate.pressed)
        {
            generate_network(&app);
        }

        draw_button(&btn_train, true, mouse_pressed);
        if (btn_train.pressed)
        {
            train_network(&app);
        }

        draw_button(&btn_save, true, mouse_pressed);
        if (btn_save.pressed)
        {
            save_network(&app, "mnist.nnw");
        }

        draw_button(&btn_load, true, mouse_pressed);
        if (btn_load.pressed)
        {
            load_network(&app, "mnist.nnw");
        }

        draw_button(&btn_reset, true, mouse_pressed);
        if (btn_reset.pressed)
        {
            reset_drawing(app.draw_buffer, 28, 28);
            if (app.prediction)
            {
                free_prediction_result(app.prediction);
                app.prediction = NULL;
            }
        }

        draw_button(&btn_random, true, mouse_pressed);
        if (btn_random.pressed)
        {
            load_random_sample(&app, app.draw_buffer, 28, 28);
            predict_digit(&app, app.draw_buffer);
        }

        // Handle drawing
        handle_drawing(draw_area, app.draw_buffer, 28, 28, draw_scale);

        // Predict on every frame when drawing
        if (IsMouseButtonDown(MOUSE_LEFT_BUTTON) && CheckCollisionPointRec(GetMousePosition(), draw_area))
        {
            predict_digit(&app, app.draw_buffer);
        }

        // Draw
        BeginDrawing();
        ClearBackground(RAYWHITE);

        // Draw buttons
        DrawRectangleRec(btn_generate.bounds, btn_generate.hovered ? DARKGRAY : LIGHTGRAY);
        DrawRectangleLinesEx(btn_generate.bounds, 2, BLACK);
        DrawText(btn_generate.label, btn_generate.bounds.x + 10, btn_generate.bounds.y + 10, 16, BLACK);

        DrawRectangleRec(btn_train.bounds, btn_train.hovered ? DARKGRAY : LIGHTGRAY);
        DrawRectangleLinesEx(btn_train.bounds, 2, BLACK);
        DrawText(btn_train.label, btn_train.bounds.x + 20, btn_train.bounds.y + 10, 16, BLACK);

        DrawRectangleRec(btn_save.bounds, btn_save.hovered ? DARKGRAY : LIGHTGRAY);
        DrawRectangleLinesEx(btn_save.bounds, 2, BLACK);
        DrawText(btn_save.label, btn_save.bounds.x + 25, btn_save.bounds.y + 10, 16, BLACK);

        DrawRectangleRec(btn_load.bounds, btn_load.hovered ? DARKGRAY : LIGHTGRAY);
        DrawRectangleLinesEx(btn_load.bounds, 2, BLACK);
        DrawText(btn_load.label, btn_load.bounds.x + 30, btn_load.bounds.y + 10, 16, BLACK);

        DrawRectangleRec(btn_reset.bounds, btn_reset.hovered ? DARKGRAY : LIGHTGRAY);
        DrawRectangleLinesEx(btn_reset.bounds, 2, BLACK);
        DrawText(btn_reset.label, btn_reset.bounds.x + 25, btn_reset.bounds.y + 10, 16, BLACK);

        DrawRectangleRec(btn_random.bounds, btn_random.hovered ? DARKGRAY : LIGHTGRAY);
        DrawRectangleLinesEx(btn_random.bounds, 2, BLACK);
        DrawText(btn_random.label, btn_random.bounds.x + 15, btn_random.bounds.y + 10, 16, BLACK);

        // Draw drawing area
        DrawText("Draw Area (28x28):", 50, 70, 16, BLACK);
        draw_drawing_area(draw_area, app.draw_buffer, 28, 28, draw_scale);

        // Draw network information panel
        DrawText("Network Information:", 400, 70, 20, BLACK);

        if (app.has_network)
        {
            char buf[256];

            snprintf(buf, sizeof(buf), "Network Status: LOADED");
            DrawText(buf, 400, 100, 16, DARKGREEN);

            snprintf(buf, sizeof(buf), "Layers: %d", app.net->size);
            DrawText(buf, 400, 130, 16, BLACK);

            snprintf(buf, sizeof(buf), "Total Connections: %d", app.connection_count);
            DrawText(buf, 400, 160, 16, BLACK);

            snprintf(buf, sizeof(buf), "Activation Functions:");
            DrawText(buf, 400, 190, 16, BLACK);

            snprintf(buf, sizeof(buf), "  Linear: %.1f%%", app.func_percentages[FUNC_LINE]);
            DrawText(buf, 420, 215, 14, BLACK);

            snprintf(buf, sizeof(buf), "  Step: %.1f%%", app.func_percentages[FUNC_STEP]);
            DrawText(buf, 420, 235, 14, BLACK);

            snprintf(buf, sizeof(buf), "  ReLU: %.1f%%", app.func_percentages[FUNC_RELU]);
            DrawText(buf, 420, 255, 14, BLACK);

            snprintf(buf, sizeof(buf), "  Sigmoid: %.1f%%", app.func_percentages[FUNC_SIGMOID]);
            DrawText(buf, 420, 275, 14, BLACK);

            snprintf(buf, sizeof(buf), "  Softsign: %.1f%%", app.func_percentages[FUNC_SOFTSIGN]);
            DrawText(buf, 420, 295, 14, BLACK);

            if (app.prediction)
            {
                snprintf(buf, sizeof(buf), "Predicted Digit: %d (Confidence: %.1f%%)", app.prediction->predicted_digit,
                         app.prediction->confidence);
                DrawText(buf, 400, 330, 18, RED);

                // Draw class probabilities
                int prob_y = 360;
                for (int i = 0; i < 10 && i < app.net->layers[app.net->size - 1].size; i++)
                {
                    snprintf(buf, sizeof(buf), "  %d: %.1f%%", i, app.prediction->class_outputs[i] * 100.0f);
                    DrawText(buf, 420, prob_y, 12, BLACK);
                    prob_y += 18;
                }
            }
        }
        else
        {
            DrawText("Network Status: NOT LOADED", 400, 100, 16, RED);
            DrawText("Generate or load a network to begin", 400, 130, 14, GRAY);
        }

        //// Draw layer architecture
        // DrawText("Layer Architecture:", 400, 380, 20, BLACK);
        // if (app.has_network)
        //{
        //     int layer_y = 410;
        //     for (int i = 0; i < app.net->size; i++)
        //     {
        //         char buf[128];
        //         snprintf(buf, sizeof(buf), "Layer %d: %d neurons", i, app.net->layers[i].size);
        //         DrawText(buf, 420, layer_y, 14, BLACK);
        //         layer_y += 25;
        //     }
        // }

        // Draw instructions
        DrawText("Instructions:", 50, 450, 20, BLACK);
        DrawText("1. Click 'Generate Network' to create a new network", 50, 480, 14, GRAY);
        DrawText("2. Click 'Train Network' to train it on the MNIST dataset", 50, 500, 14, GRAY);
        DrawText("3. Draw a digit in the draw area - the network will predict it in real-time", 50, 520, 14, GRAY);
        DrawText("4. Use 'Random Sample' to test with dataset samples", 50, 540, 14, GRAY);
        DrawText("5. Click 'Save/Load' to persist your trained network", 50, 560, 14, GRAY);

        EndDrawing();
    }

    // Cleanup
    if (app.draw_buffer)
        free(app.draw_buffer);
    if (app.func_counts)
        free(app.func_counts);
    if (app.func_percentages)
        free(app.func_percentages);
    if (app.mnist_split)
        free_train_test_split(app.mnist_split);
    if (app.net && app.has_network)
        free_neural_net(app.net);
    if (app.prediction)
        free_prediction_result(app.prediction);

    CloseWindow();

    return 0;
}
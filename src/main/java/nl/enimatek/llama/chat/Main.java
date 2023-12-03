package nl.enimatek.llama.chat;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;

import de.kherud.llama.InferenceParameters;
import de.kherud.llama.LlamaModel;
import de.kherud.llama.ModelParameters;
import lombok.extern.java.Log;

@Log
public class Main {
    public static void main(String... args) throws IOException {
        LlamaModel.setLogger((level, message) -> log.info(message));
        var modelParams = new ModelParameters.Builder().setNGpuLayers(43).build();
        InferenceParameters inferParams = new InferenceParameters.Builder()
                .setTemperature(0.7f)
                .setPenalizeNl(true)
                // .setNProbs(10)
                .setMirostat(InferenceParameters.MiroStat.V2)
                .setAntiPrompt("\n").build();

        String modelPath = "llama-2-7b-chat.Q2_K.gguf";
        String system = "This is a conversation between User and Llama, a friendly chatbot.\n" +
                "requests immediately and with precision.\n";

        BufferedReader reader = new BufferedReader(new InputStreamReader(System.in, StandardCharsets.UTF_8));

        try (LlamaModel model = new LlamaModel(modelPath, modelParams)) {
            var prompt = new StringBuilder();
            prompt.append(system);
            while (true) {
                prompt.append("User: ");
                String input = reader.readLine();
                prompt.append(input);                
                prompt.append(" ").append("Llama: ");
                for (LlamaModel.Output output : model.generate(prompt.toString(), inferParams)) {
                    System.out.print(output.text);
                    prompt.append(output.text);
                }
            }
        }
    }
}
import os
import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from glob import glob
from tqdm import tqdm
from collections import defaultdict
import warnings

warnings.filterwarnings('ignore')

# Configuración
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAMPLE_RATE = 22050
MAX_LEN = 3  # segundos
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 3e-4
DATA_DIR = r"C:\Users\saray\Downloads\oyopf\all-sample-des"  # Cambiar según tu ruta

# 1. Procesamiento de Audio
class AudioProcessor:
    def __init__(self, sample_rate=22050, n_mels=128):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=2048,
            hop_length=512,
            n_mels=n_mels,
            power=2
        )
        self.db_transform = torchaudio.transforms.AmplitudeToDB()

    def process(self, waveform, sample_rate):
        if sample_rate != self.sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.sample_rate)
            waveform = resampler(waveform)
        
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        target_samples = int(self.sample_rate * MAX_LEN)
        if waveform.shape[1] > target_samples:
            waveform = waveform[:, :target_samples]
        else:
            pad_amount = target_samples - waveform.shape[1]
            waveform = F.pad(waveform, (0, pad_amount))
        
        waveform = waveform / waveform.abs().max()
        mel = self.mel_transform(waveform)
        mel_db = self.db_transform(mel)
        mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-8)
        return mel_db

# 2. Dataset
class InstrumentDataset(Dataset):
    def __init__(self, data_dir, augment=False, max_samples_per_class=200):
        self.processor = AudioProcessor(SAMPLE_RATE)
        self.augment = augment
        self.samples = []
        self.label_map = {}
        self.inverse_map = {}
        
        class_dirs = sorted(glob(os.path.join(data_dir, "*")))
        for class_idx, class_dir in enumerate(class_dirs):
            class_name = os.path.basename(class_dir)
            self.label_map[class_idx] = class_name
            self.inverse_map[class_name] = class_idx
            
            audio_files = glob(os.path.join(class_dir, "*.mp3")) + glob(os.path.join(class_dir, "*.wav"))
            audio_files = audio_files[:max_samples_per_class]
            
            for audio_file in tqdm(audio_files, desc=f"Cargando {class_name}"):
                self.samples.append((audio_file, class_idx))
        
        print(f"\nDataset cargado: {len(self.samples)} muestras, {len(self.label_map)} clases")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        audio_path, label = self.samples[idx]
        try:
            waveform, sample_rate = torchaudio.load(audio_path)
            spec = self.processor.process(waveform, sample_rate)
            
            if self.augment and np.random.random() < 0.5:
                spec = torch.roll(spec, shifts=np.random.randint(-10, 10), dims=2)
            
            if spec.dim() == 2:
                spec = spec.unsqueeze(0)
            return spec, label
            
        except Exception as e:
            print(f"Error procesando {audio_path}: {str(e)}")
            dummy = torch.zeros((1, self.processor.n_mels, int(SAMPLE_RATE * MAX_LEN / 512) + 1))
            return dummy, 0

    def get_class_weights(self):
        counts = np.bincount([label for _, label in self.samples])
        counts = np.where(counts == 0, 1, counts)
        weights = 1. / counts
        weights = weights / weights.sum() * len(weights)
        return torch.tensor(weights, dtype=torch.float32)

# 3. Modelo CNN
class InstrumentCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.2),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.3),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.4),
            
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

# 4. Visualización
class TrainingVisualizer:
    def __init__(self, label_map):
        self.label_map = label_map
        self.train_loss = []
        self.val_loss = []
        self.train_acc = []
        self.val_acc = []
        
    def update(self, epoch, tr_loss, val_loss, tr_acc, val_acc, model, val_loader):
        self.train_loss.append(tr_loss)
        self.val_loss.append(val_loss)
        self.train_acc.append(tr_acc)
        self.val_acc.append(val_acc)
        
        if epoch % 5 == 0 or epoch == EPOCHS - 1:
            self._plot_metrics()
            self._plot_confusion_matrix(model, val_loader)
    
    def _plot_metrics(self):
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(self.train_loss, label='Entrenamiento')
        plt.plot(self.val_loss, label='Validación')
        plt.title('Evolución de la Pérdida')
        plt.xlabel('Época')
        plt.ylabel('Pérdida')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(self.train_acc, label='Entrenamiento')
        plt.plot(self.val_acc, label='Validación')
        plt.title('Evolución de la Precisión')
        plt.xlabel('Época')
        plt.ylabel('Precisión')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('training_metrics.png')
        plt.close()
    
    def _plot_confusion_matrix(self, model, loader):
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in loader:
                inputs = inputs.to(DEVICE)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.label_map.values(),
                    yticklabels=self.label_map.values())
        plt.title('Matriz de Confusión')
        plt.xlabel('Predicciones')
        plt.ylabel('Valores Verdaderos')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png')
        plt.close()
        
        print("\nReporte de Clasificación:")
        print(classification_report(all_labels, all_preds, target_names=self.label_map.values()))

# 5. Funciones de Entrenamiento
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, label_map):
    visualizer = TrainingVisualizer(label_map)
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_corrects = 0
        
        for inputs, labels in train_loader:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        
        train_loss = running_loss / len(train_loader.dataset)
        train_acc = running_corrects.double() / len(train_loader.dataset)
        
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                _, preds = torch.max(outputs, 1)
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)
        
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_corrects.double() / len(val_loader.dataset)
        
        if scheduler:
            scheduler.step(val_loss)
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
        
        visualizer.update(epoch, train_loss, val_loss, train_acc.item(), val_acc.item(), model, val_loader)
        
        print(f'Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}')
    
    return model

def create_dataloaders(data_dir, batch_size=32, val_split=0.2):
    full_dataset = InstrumentDataset(data_dir, augment=True)
    
    indices = list(range(len(full_dataset)))
    labels = [full_dataset.samples[i][1] for i in indices]
    
    train_indices, val_indices = train_test_split(indices, test_size=val_split, stratify=labels)
    
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
    
    train_labels = [full_dataset.samples[i][1] for i in train_indices]
    class_weights = full_dataset.get_class_weights()
    sample_weights = class_weights[train_labels]
    
    if (sample_weights <= 0).any():
        sample_weights = torch.clamp(sample_weights, min=1e-8)
    
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    return train_loader, val_loader, full_dataset.label_map

# 6. Clasificación de Audios Nuevos
def predict_audio(file_path, model_path='best_model.pth', threshold=0.6, show_spectrogram=True):
    if not hasattr(predict_audio, 'label_map'):
        _, _, predict_audio.label_map = create_dataloaders(DATA_DIR, BATCH_SIZE)
        predict_audio.model = InstrumentCNN(len(predict_audio.label_map)).to(DEVICE)
        predict_audio.model.load_state_dict(torch.load(model_path))
        predict_audio.model.eval()
        predict_audio.processor = AudioProcessor(SAMPLE_RATE)
    
    try:
        waveform, sample_rate = torchaudio.load(file_path)
        spec = predict_audio.processor.process(waveform, sample_rate)
        spec = spec.unsqueeze(0).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            outputs = predict_audio.model(spec)
            probs = F.softmax(outputs, dim=1)
            conf, pred = torch.max(probs, 1)
            conf = conf.item()
            pred_class = predict_audio.label_map[pred.item()]
        
        print(f"\n Audio analizado: {os.path.basename(file_path)}")
        
        if conf >= threshold:
            print(f"🎵 Predicción: {pred_class} (Confianza: {conf:.2%})")
        else:
            print(f" Predicción incierta: {pred_class} (Confianza: {conf:.2%} < {threshold:.0%})")
        
        print("\nDistribución de probabilidades:")
        for i, prob in enumerate(probs.squeeze().cpu().numpy()):
            print(f"- {predict_audio.label_map[i]}: {prob:.2%}")
        
        if show_spectrogram:
            plt.figure(figsize=(10, 4))
            plt.imshow(spec.squeeze().cpu().numpy(), aspect='auto', origin='lower')
            plt.title(f"Espectrograma | Pred: {pred_class} ({conf:.2%})")
            plt.colorbar()
            plt.tight_layout()
            plt.show()
        
        return pred_class, conf
    
    except Exception as e:
        print(f"Error procesando el audio: {str(e)}")
        return None, None

# 7. Función Principal y Menú
def main():
    print("=== CLASIFICADOR DE INSTRUMENTOS MUSICALES ===")
    print(f"Dispositivo: {DEVICE}")
    
    print("\nCargando dataset...")
    train_loader, val_loader, label_map = create_dataloaders(DATA_DIR, BATCH_SIZE)
    
    model = InstrumentCNN(len(label_map)).to(DEVICE)
    print(f"\nModelo creado con {len(label_map)} clases")
    print(f"Total parámetros: {sum(p.numel() for p in model.parameters()):,}")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    print("\nIniciando entrenamiento...")
    model = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, EPOCHS, label_map)
    
    print("\nEntrenamiento completado!")
    print(f"Mejores pesos guardados en: best_model.pth")

if __name__ == "__main__":
    print("=== MENÚ ===")
    print("1. Entrenar modelo")
    print("2. Clasificar un audio")
    print("3. Salir")

    choice = input("Selecciona una opción: ")
    if choice == "1":
        main()
    elif choice == "2":
        path = input("Ruta del archivo de audio: ")
        if os.path.exists(path):
            predict_audio(path)
        else:
            print(" El archivo no existe. Verifica la ruta.")
    elif choice == "3":
        print("¡Hasta luego!")
    else:
        print("Opción no válida.")
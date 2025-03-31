import { 
  users, type User, type InsertUser,
  hyperparameters, type Hyperparameter, type InsertHyperparameter,
  frameworks, type Framework, type InsertFramework
} from "@shared/schema";

export interface IStorage {
  // User methods
  getUser(id: number): Promise<User | undefined>;
  getUserByUsername(username: string): Promise<User | undefined>;
  createUser(user: InsertUser): Promise<User>;
  
  // Hyperparameter methods
  getAllHyperparameters(): Promise<Hyperparameter[]>;
  getHyperparameterByKey(paramKey: string): Promise<Hyperparameter | undefined>;
  getHyperparametersByFramework(framework: string): Promise<Hyperparameter[]>;
  createHyperparameter(hyperparameter: InsertHyperparameter): Promise<Hyperparameter>;
  
  // Framework methods
  getAllFrameworks(): Promise<Framework[]>;
  getFrameworkByName(name: string): Promise<Framework | undefined>;
  createFramework(framework: InsertFramework): Promise<Framework>;
}

export class MemStorage implements IStorage {
  private users: Map<number, User>;
  private hyperparams: Map<number, Hyperparameter>;
  private frameworksMap: Map<number, Framework>;
  private userCurrentId: number;
  private hyperparamCurrentId: number;
  private frameworkCurrentId: number;

  constructor() {
    this.users = new Map();
    this.hyperparams = new Map();
    this.frameworksMap = new Map();
    this.userCurrentId = 1;
    this.hyperparamCurrentId = 1;
    this.frameworkCurrentId = 1;
    
    // Initialize with some default frameworks
    this.initDefaultFrameworks();
    // Initialize with some default hyperparameters
    this.initDefaultHyperparameters();
  }
  
  // Initialize some popular ML frameworks
  private initDefaultFrameworks() {
    const defaultFrameworks: InsertFramework[] = [
      { name: "TensorFlow", language: "Python", logoUrl: "tensorflow.svg" },
      { name: "PyTorch", language: "Python", logoUrl: "pytorch.svg" },
      { name: "Keras", language: "Python", logoUrl: "keras.svg" },
      { name: "Scikit-learn", language: "Python", logoUrl: "scikit-learn.svg" },
      { name: "TensorFlow.js", language: "JavaScript", logoUrl: "tensorflow.svg" },
      { name: "Brain.js", language: "JavaScript", logoUrl: "brainjs.svg" },
      { name: "Hugging Face", language: "Python", logoUrl: "huggingface.svg" }
    ];
    
    defaultFrameworks.forEach(framework => {
      this.createFramework(framework);
    });
  }
  
  // Initialize some common hyperparameters
  private initDefaultHyperparameters() {
    const defaultHyperparameters: InsertHyperparameter[] = [
      {
        name: "Learning Rate",
        paramKey: "learning_rate",
        description: "The rate at which the model learns from each batch of data.",
        framework: "TensorFlow",
        impact: "high",
        defaultValue: "0.001",
        alternatives: JSON.stringify([
          { value: "0.0001", description: "Use for fine-tuning or when training is unstable", type: "lower" },
          { value: "0.01", description: "Faster learning but may overshoot", type: "higher" },
          { value: "1e-5", description: "Very slow learning for final stages", type: "extreme" }
        ])
      },
      {
        name: "Batch Size",
        paramKey: "batch_size",
        description: "The number of training examples used in one iteration.",
        framework: "PyTorch",
        impact: "medium",
        defaultValue: "32",
        alternatives: JSON.stringify([
          { value: "16", description: "Smaller batch size for better generalization", type: "lower" },
          { value: "64", description: "Larger batch size for faster training", type: "higher" },
          { value: "128", description: "Much larger batch size for large models", type: "extreme" }
        ])
      },
      {
        name: "Epochs",
        paramKey: "epochs",
        description: "The number of complete passes through the training dataset.",
        framework: "Keras",
        impact: "medium",
        defaultValue: "10",
        alternatives: JSON.stringify([
          { value: "5", description: "Fewer epochs to prevent overfitting", type: "lower" },
          { value: "20", description: "More epochs for complex datasets", type: "higher" },
          { value: "100", description: "Many epochs with early stopping", type: "extreme" }
        ])
      },
      {
        name: "Dropout Rate",
        paramKey: "dropout",
        description: "The fraction of input units to drop during training to prevent overfitting.",
        framework: "TensorFlow",
        impact: "medium",
        defaultValue: "0.2",
        alternatives: JSON.stringify([
          { value: "0.1", description: "Lower dropout for simpler models", type: "lower" },
          { value: "0.5", description: "Higher dropout for complex models", type: "higher" },
          { value: "0.8", description: "Extreme dropout for very deep networks", type: "extreme" }
        ])
      }
    ];
    
    defaultHyperparameters.forEach(hyperparameter => {
      this.createHyperparameter(hyperparameter);
    });
  }

  // User methods
  async getUser(id: number): Promise<User | undefined> {
    return this.users.get(id);
  }

  async getUserByUsername(username: string): Promise<User | undefined> {
    return Array.from(this.users.values()).find(
      (user) => user.username === username,
    );
  }

  async createUser(insertUser: InsertUser): Promise<User> {
    const id = this.userCurrentId++;
    const user: User = { ...insertUser, id };
    this.users.set(id, user);
    return user;
  }
  
  // Hyperparameter methods
  async getAllHyperparameters(): Promise<Hyperparameter[]> {
    return Array.from(this.hyperparams.values());
  }
  
  async getHyperparameterByKey(paramKey: string): Promise<Hyperparameter | undefined> {
    return Array.from(this.hyperparams.values()).find(
      (hyperparam) => hyperparam.paramKey === paramKey
    );
  }
  
  async getHyperparametersByFramework(framework: string): Promise<Hyperparameter[]> {
    return Array.from(this.hyperparams.values()).filter(
      (hyperparam) => hyperparam.framework === framework
    );
  }
  
  async createHyperparameter(insertHyperparameter: InsertHyperparameter): Promise<Hyperparameter> {
    const id = this.hyperparamCurrentId++;
    // Ensure defaultValue is not undefined by providing null as fallback
    const defaultValue = insertHyperparameter.defaultValue ?? null;
    // Ensure alternatives is not undefined
    const alternatives = insertHyperparameter.alternatives ?? null;
    
    const hyperparameter: Hyperparameter = { 
      ...insertHyperparameter, 
      id,
      defaultValue,
      alternatives 
    };
    
    this.hyperparams.set(id, hyperparameter);
    return hyperparameter;
  }
  
  // Framework methods
  async getAllFrameworks(): Promise<Framework[]> {
    return Array.from(this.frameworksMap.values());
  }
  
  async getFrameworkByName(name: string): Promise<Framework | undefined> {
    return Array.from(this.frameworksMap.values()).find(
      (framework) => framework.name === name
    );
  }
  
  async createFramework(insertFramework: InsertFramework): Promise<Framework> {
    const id = this.frameworkCurrentId++;
    // Ensure logoUrl is not undefined by providing null as fallback
    const logoUrl = insertFramework.logoUrl ?? null;
    
    const framework: Framework = { 
      ...insertFramework, 
      id,
      logoUrl 
    };
    
    this.frameworksMap.set(id, framework);
    return framework;
  }
}

export const storage = new MemStorage();

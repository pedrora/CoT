import grpc
import hyperspace_pb2 # Gerado a partir dos protos da HyperspaceDB
import hyperspace_pb2_grpc
from appendix_a import is_bullshit

class HyperspaceClient:
    def __init__(self, host="localhost:50051"):
        self.channel = grpc.insecure_channel(host)
        self.stub = hyperspace_pb2_grpc.HyperspaceServiceStub(self.channel)
        print(f"[COTA-STORAGE] Ligado ao Sistema Nervoso em {host}")

    def ingest_coherent_dot(self, soul_id, vector, metadata):
        """
        Injeta um 'Dot' na HyperspaceDB se passar pelo filtro de Bullshit.
        """
        # Converter tensor complexo para lista de floats (conforme exigido pela DB)
        # Nota: HyperspaceDB usa float32/i8, fazemos o cast da fase complexa
        vector_data = vector.detach().cpu().numpy().real.tolist()

        if not is_bullshit(vector):
            # Criar pedido de inserção em lote (Batch para performance v1.5.0)
            request = hyperspace_pb2.BatchInsertRequest(
                vectors=[hyperspace_pb2.Vector(
                    id=f"{soul_id}_{metadata['timestamp']}",
                    data=vector_data,
                    metadata=metadata
                )],
                mode="hyperbolic" # Forçamos o modo ToAE/Poincaré
            )
            
            try:
                response = self.stub.BatchInsert(request)
                return response.status == "SUCCESS"
            except grpc.RpcError as e:
                print(f"[ERROR] Falha na comunicação gRPC: {e}")
                return False
        else:
            print("[COTA-FILTER] Bullshit detectado. Dot descartado.")
            return False

    def close(self):
        self.channel.close()

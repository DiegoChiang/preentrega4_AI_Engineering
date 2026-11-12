# Diagnóstico de Kubernetes

## Pods en CrashLoopBackOff

El diagnóstico comienza con `kubectl describe pod` para revisar eventos, probes y límites de
recursos. Luego se consultan los logs actuales y anteriores mediante `kubectl logs` y
`kubectl logs --previous`. Un error de configuración requiere comparar el ConfigMap, los Secrets y
las variables del deployment con la revisión estable.

No se deben reiniciar todos los pods al mismo tiempo. Primero se identifica si el fallo afecta una
réplica, una revisión del deployment o todo el nodo. Si existe presión de memoria, se inspeccionan
los eventos OOMKilled, el consumo real y los límites configurados antes de elevar recursos.

## Problemas de red

Para errores de conectividad se valida la resolución DNS desde un pod de diagnóstico, los endpoints
del Service y las NetworkPolicies. Un Service sin endpoints indica que sus selectores no coinciden
con las etiquetas de los pods o que ninguna réplica está Ready.

## Recuperación

Un rollback se realiza con `kubectl rollout undo`. La recuperación se valida observando readiness,
tasa de errores, latencia y saturación durante quince minutos. Los eventos, comandos y cambios de
configuración quedan registrados en la bitácora del incidente.
